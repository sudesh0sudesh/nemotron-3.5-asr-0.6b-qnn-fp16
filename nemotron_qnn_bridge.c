/* SPDX-License-Identifier: Apache-2.0 */
/* Minimal QNN context-binary bridge for the Nemotron V75 prototype.
 * Owns all runtime handles and keeps binary metadata alive until close.
 * CPU preprocessing/control is outside this library. HTP is the only backend.
 */
#define _POSIX_C_SOURCE 200809L
#include <dlfcn.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "QnnInterface.h"
#include "System/QnnSystemInterface.h"
#include "HTP/QnnHtpDevice.h"
#include "HTP/QnnHtpProfile.h"

typedef struct {
    void *htp_lib, *system_lib, *binary;
    size_t binary_size;
    const QNN_INTERFACE_VER_TYPE *api;
    const QNN_SYSTEM_INTERFACE_VER_TYPE *sys;
    Qnn_LogHandle_t log;
    Qnn_BackendHandle_t backend;
    Qnn_ContextHandle_t context;
    Qnn_GraphHandle_t graph;
    QnnSystemContext_Handle_t system_context;
    Qnn_Tensor_t *inputs, *outputs;
    uint32_t input_count, output_count;
    QnnHtpDevice_Infrastructure_t *infra;
    uint32_t power_id;
    bool power_created;
} Model;

#define FIELD(t, member) ((t)->version == QNN_TENSOR_VERSION_1 ? (t)->v1.member : (t)->v2.member)
void nq_close(Model *m);
static void logger(const char *fmt,QnnLog_Level_t level,uint64_t timestamp,va_list args) {
    (void)level;(void)timestamp;vfprintf(stderr,fmt,args);fputc('\n',stderr);
}
static Qnn_Tensor_t *tensor(Model *m,int output,uint32_t index) {
    if(!m || index >= (output?m->output_count:m->input_count))return NULL;
    return &(output?m->outputs:m->inputs)[index];
}
uint32_t nq_count(Model *m,int output) {return m?(output?m->output_count:m->input_count):0;}
const char *nq_name(Model *m,int output,uint32_t index) {
    Qnn_Tensor_t *t=tensor(m,output,index);return t?FIELD(t,name):NULL;
}
uint32_t nq_rank(Model *m,int output,uint32_t index) {
    Qnn_Tensor_t *t=tensor(m,output,index);return t?FIELD(t,rank):0;
}
uint32_t nq_dimension(Model *m,int output,uint32_t index,uint32_t axis) {
    Qnn_Tensor_t *t=tensor(m,output,index);return t && axis<FIELD(t,rank)?FIELD(t,dimensions)[axis]:0;
}
const char *nq_dtype(Model *m,int output,uint32_t index) {
    Qnn_Tensor_t *t=tensor(m,output,index);if(!t)return NULL;
    switch(FIELD(t,dataType)) {
        case QNN_DATATYPE_FLOAT_32:return "float32";
        case QNN_DATATYPE_FLOAT_16:return "float16";
        case QNN_DATATYPE_INT_32:return "int32";
        case QNN_DATATYPE_UINT_32:return "uint32";
        case QNN_DATATYPE_UFIXED_POINT_8:case QNN_DATATYPE_UINT_8:return "uint8";
        case QNN_DATATYPE_SFIXED_POINT_8:case QNN_DATATYPE_INT_8:return "int8";
        case QNN_DATATYPE_UFIXED_POINT_16:case QNN_DATATYPE_UINT_16:return "uint16";
        case QNN_DATATYPE_SFIXED_POINT_16:case QNN_DATATYPE_INT_16:return "int16";
        default:return "unsupported";
    }
}
static size_t tensor_bytes(Qnn_Tensor_t *t) {
    size_t bytes=0;
    switch(FIELD(t,dataType)) {
        case QNN_DATATYPE_FLOAT_32:case QNN_DATATYPE_INT_32:case QNN_DATATYPE_UINT_32:bytes=4;break;
        case QNN_DATATYPE_FLOAT_16:case QNN_DATATYPE_UFIXED_POINT_16:case QNN_DATATYPE_UINT_16:case QNN_DATATYPE_SFIXED_POINT_16:case QNN_DATATYPE_INT_16:bytes=2;break;
        case QNN_DATATYPE_UFIXED_POINT_8:case QNN_DATATYPE_UINT_8:case QNN_DATATYPE_SFIXED_POINT_8:case QNN_DATATYPE_INT_8:bytes=1;break;
        default:return 0;
    }
    for(uint32_t i=0;i<FIELD(t,rank);++i) {
        uint32_t d=FIELD(t,dimensions)[i];if(!d||bytes>UINT32_MAX/d)return 0;bytes*=d;
    }
    return bytes;
}
size_t nq_bytes(Model *m,int output,uint32_t index) {
    Qnn_Tensor_t *t=tensor(m,output,index);return t?tensor_bytes(t):0;
}
Model *nq_open(const char *path,int performance,char *error,size_t error_size) {
    Model *m=calloc(1,sizeof(*m));if(!m)return NULL;
    Qnn_ErrorHandle_t rc=QNN_SUCCESS;const char *stage="load libraries";
#define CHECK(label,call) do {stage=label;rc=(call);if(rc!=QNN_SUCCESS)goto fail;} while(0)
    m->htp_lib=dlopen("/usr/lib/libQnnHtp.so",RTLD_NOW|RTLD_LOCAL);
    m->system_lib=dlopen("/usr/lib/libQnnSystem.so",RTLD_NOW|RTLD_LOCAL);
    if(!m->htp_lib||!m->system_lib)goto fail;
    Qnn_ErrorHandle_t (*get_api)(const QnnInterface_t ***,uint32_t *)=dlsym(m->htp_lib,"QnnInterface_getProviders");
    Qnn_ErrorHandle_t (*get_sys)(const QnnSystemInterface_t ***,uint32_t *)=dlsym(m->system_lib,"QnnSystemInterface_getProviders");
    if(!get_api||!get_sys)goto fail;
    const QnnInterface_t **apis=NULL;const QnnSystemInterface_t **systems=NULL;uint32_t count=0;
    CHECK("QNN providers",get_api(&apis,&count));if(!count)goto fail;
    if(apis[0]->apiVersion.coreApiVersion.major!=QNN_API_VERSION_MAJOR)goto fail;
    m->api=&apis[0]->QNN_INTERFACE_VER_NAME;
    CHECK("system providers",get_sys(&systems,&count));if(!count)goto fail;
    m->sys=&systems[0]->QNN_SYSTEM_INTERFACE_VER_NAME;
    stage="map context file";
    int fd=open(path,O_RDONLY);if(fd<0)goto fail;
    struct stat st;if(fstat(fd,&st)!=0||st.st_size<=0){close(fd);goto fail;}
    m->binary_size=(size_t)st.st_size;m->binary=mmap(NULL,m->binary_size,PROT_READ,MAP_PRIVATE,fd,0);close(fd);
    if(m->binary==MAP_FAILED){m->binary=NULL;goto fail;}
    CHECK("metadata context",m->sys->systemContextCreate(&m->system_context));
    const QnnSystemContext_BinaryInfo_t *info=NULL;Qnn_ContextBinarySize_t info_size=0;
    CHECK("binary metadata",m->sys->systemContextGetBinaryInfo(m->system_context,m->binary,m->binary_size,&info,&info_size));
    QnnSystemContext_GraphInfo_t *graphs=NULL;uint32_t graph_count=0;
    switch(info->version) {
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1:graphs=info->contextBinaryInfoV1.graphs;graph_count=info->contextBinaryInfoV1.numGraphs;break;
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2:graphs=info->contextBinaryInfoV2.graphs;graph_count=info->contextBinaryInfoV2.numGraphs;break;
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3:
            graphs=info->contextBinaryInfoV3.graphs;graph_count=info->contextBinaryInfoV3.numGraphs;
            if(info->contextBinaryInfoV3.socModel && info->contextBinaryInfoV3.socModel!=82){stage="context is not for SoC 82";goto fail;}
            break;
        default:stage="unsupported metadata version";goto fail;
    }
    if(graph_count!=1){stage="expected exactly one graph";goto fail;}
    const char *graph_name=NULL;Qnn_Tensor_t *in=NULL,*out=NULL;
#define GRAPH_FIELDS(v) do {graph_name=graphs[0].v.graphName;m->input_count=graphs[0].v.numGraphInputs;m->output_count=graphs[0].v.numGraphOutputs;in=graphs[0].v.graphInputs;out=graphs[0].v.graphOutputs;}while(0)
    switch(graphs[0].version) {
        case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1:GRAPH_FIELDS(graphInfoV1);break;
        case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2:GRAPH_FIELDS(graphInfoV2);break;
        case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3:GRAPH_FIELDS(graphInfoV3);break;
        default:stage="unsupported graph metadata version";goto fail;
    }
    m->inputs=calloc(m->input_count,sizeof(Qnn_Tensor_t));m->outputs=calloc(m->output_count,sizeof(Qnn_Tensor_t));
    if(!m->inputs||!m->outputs)goto fail;
    memcpy(m->inputs,in,m->input_count*sizeof(*in));memcpy(m->outputs,out,m->output_count*sizeof(*out));
    CHECK("logger",m->api->logCreate(logger,QNN_LOG_LEVEL_WARN,&m->log));
    CHECK("HTP backend",m->api->backendCreate(m->log,NULL,&m->backend));
    CHECK("restore context",m->api->contextCreateFromBinary(m->backend,NULL,NULL,m->binary,m->binary_size,&m->context,NULL));
    CHECK("retrieve graph",m->api->graphRetrieve(m->context,graph_name,&m->graph));
    if(performance) {
        CHECK("performance infrastructure",m->api->deviceGetInfrastructure((const QnnDevice_Infrastructure_t*)&m->infra));
        if(!m->infra||m->infra->infraType!=QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF)goto fail;
        CHECK("power id",m->infra->perfInfra.createPowerConfigId(0,0,&m->power_id));m->power_created=true;
        QnnHtpPerfInfrastructure_PowerConfig_t p={0};p.option=QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
        QnnHtpPerfInfrastructure_DcvsV3_t *v=&p.dcvsV3Config;
        v->contextId=m->power_id;v->setDcvsEnable=1;v->dcvsEnable=0;v->powerMode=QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
        v->setSleepLatency=1;v->sleepLatency=40;v->setSleepDisable=1;v->sleepDisable=1;
        v->setBusParams=1;v->busVoltageCornerMin=v->busVoltageCornerTarget=v->busVoltageCornerMax=DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        v->setCoreParams=1;v->coreVoltageCornerMin=v->coreVoltageCornerTarget=v->coreVoltageCornerMax=DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        const QnnHtpPerfInfrastructure_PowerConfig_t *settings[]={&p,NULL};
        CHECK("DCVS performance vote",m->infra->perfInfra.setPowerConfig(m->power_id,settings));
    }
    return m;
fail:
    if(error && error_size)snprintf(error,error_size,"%s: QNN error %llu",stage,(unsigned long long)rc);
    nq_close(m);return NULL;
}
static void event_time(Model *m,QnnProfile_EventId_t id,unsigned depth,double *us,bool *found) {
    if(depth>16)return;
    QnnProfile_EventData_t event=QNN_PROFILE_EVENT_DATA_INIT;
    if(m->api->profileGetEventData(id,&event)!=QNN_SUCCESS)return;
    if(event.type==QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_ACCEL_TIME_MICROSEC && event.unit==QNN_PROFILE_EVENTUNIT_MICROSEC){*us+=event.value;*found=true;}
    const QnnProfile_EventId_t *children=NULL;uint32_t count=0;
    if(m->api->profileGetSubEvents(id,&children,&count)==QNN_SUCCESS)
        for(uint32_t i=0;i<count;++i)event_time(m,children[i],depth+1,us,found);
}
static int bind(Qnn_Tensor_t *t,void *buffer,size_t size) {
    if(!buffer||size!=tensor_bytes(t)||size>UINT32_MAX)return -1;
    if(t->version==QNN_TENSOR_VERSION_1){t->v1.memType=QNN_TENSORMEMTYPE_RAW;t->v1.clientBuf=(Qnn_ClientBuffer_t){buffer,(uint32_t)size};}
    else if(t->version==QNN_TENSOR_VERSION_2){t->v2.memType=QNN_TENSORMEMTYPE_RAW;t->v2.clientBuf=(Qnn_ClientBuffer_t){buffer,(uint32_t)size};}
    else return -1;
    return 0;
}
int nq_execute(Model *m,void **inputs,size_t *input_sizes,void **outputs,size_t *output_sizes,double *accelerator_us) {
    if(!m||!inputs||!outputs||!input_sizes||!output_sizes||!accelerator_us)return -1;
    for(uint32_t i=0;i<m->input_count;++i)if(bind(&m->inputs[i],inputs[i],input_sizes[i]))return -2;
    for(uint32_t i=0;i<m->output_count;++i)if(bind(&m->outputs[i],outputs[i],output_sizes[i]))return -3;
    Qnn_ProfileHandle_t profile=NULL;Qnn_ErrorHandle_t rc=m->api->profileCreate(m->backend,QNN_PROFILE_LEVEL_BASIC,&profile);
    if(rc)return (int)rc;
    rc=m->api->graphExecute(m->graph,m->inputs,m->input_count,m->outputs,m->output_count,profile,NULL);
    *accelerator_us=0;bool found=false;
    if(rc==QNN_SUCCESS){const QnnProfile_EventId_t *ids=NULL;uint32_t count=0;
        if(m->api->profileGetEvents(profile,&ids,&count)==QNN_SUCCESS)
            for(uint32_t i=0;i<count;++i)event_time(m,ids[i],0,accelerator_us,&found);
    }
    m->api->profileFree(profile);if(!found)*accelerator_us=-1;
    return (int)rc;
}
void nq_close(Model *m) {
    if(!m)return;
    if(m->power_created)m->infra->perfInfra.destroyPowerConfigId(m->power_id);
    if(m->context)m->api->contextFree(m->context,NULL);
    if(m->backend)m->api->backendFree(m->backend);
    if(m->log)m->api->logFree(m->log);
    if(m->system_context)m->sys->systemContextFree(m->system_context);
    free(m->inputs);free(m->outputs);
    if(m->binary)munmap(m->binary,m->binary_size);
    if(m->htp_lib)dlclose(m->htp_lib);
    if(m->system_lib)dlclose(m->system_lib);
    free(m);
}
