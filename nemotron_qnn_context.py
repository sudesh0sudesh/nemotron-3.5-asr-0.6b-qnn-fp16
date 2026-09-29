# SPDX-License-Identifier: Apache-2.0
"""NumPy interface to the native HTP-only context-binary bridge."""
import ctypes as C
import time
from pathlib import Path
import numpy as np

class QnnContext:
    """Single-threaded context; tensor dtypes are explicit and never quantized implicitly."""
    def __init__(self,binary,library=None,performance=True):
        self.lib=C.CDLL(str(library or Path(__file__).resolve().with_name('libnemotron_qnn_bridge.so')));self.handle=None;self.samples=[]
        lib=self.lib
        lib.nq_open.argtypes=[C.c_char_p,C.c_int,C.c_char_p,C.c_size_t];lib.nq_open.restype=C.c_void_p
        lib.nq_close.argtypes=[C.c_void_p];lib.nq_close.restype=None
        lib.nq_count.argtypes=[C.c_void_p,C.c_int];lib.nq_count.restype=C.c_uint32
        for fn in ['nq_name','nq_dtype']:
            getattr(lib,fn).argtypes=[C.c_void_p,C.c_int,C.c_uint32];getattr(lib,fn).restype=C.c_char_p
        for fn in ['nq_rank','nq_bytes']:
            getattr(lib,fn).argtypes=[C.c_void_p,C.c_int,C.c_uint32]
        lib.nq_rank.restype=C.c_uint32;lib.nq_bytes.restype=C.c_size_t
        lib.nq_dimension.argtypes=[C.c_void_p,C.c_int,C.c_uint32,C.c_uint32];lib.nq_dimension.restype=C.c_uint32
        lib.nq_execute.argtypes=[C.c_void_p,C.POINTER(C.c_void_p),C.POINTER(C.c_size_t),C.POINTER(C.c_void_p),C.POINTER(C.c_size_t),C.POINTER(C.c_double)]
        lib.nq_execute.restype=C.c_int
        error=C.create_string_buffer(1024)
        begin=time.perf_counter();self.handle=lib.nq_open(str(binary).encode(),int(performance),error,len(error));self.initialization_ms=(time.perf_counter()-begin)*1000
        if not self.handle:raise RuntimeError(error.value.decode())
        try:
            self.inputs=self._describe(False);self.outputs=self._describe(True)
        except Exception:
            self.close();raise
    def _describe(self,output):
        result=[]
        for i in range(self.lib.nq_count(self.handle,int(output))):
            dtype=np.dtype(self.lib.nq_dtype(self.handle,int(output),i).decode())
            shape=tuple(self.lib.nq_dimension(self.handle,int(output),i,j) for j in range(self.lib.nq_rank(self.handle,int(output),i)))
            count=int(np.prod(shape))*dtype.itemsize
            assert count==self.lib.nq_bytes(self.handle,int(output),i)
            result.append({'name':self.lib.nq_name(self.handle,int(output),i).decode(),'dtype':dtype.name,'shape':shape,'bytes':count})
        return result
    def run(self,values):
        if not self.handle:raise RuntimeError('Context is closed')
        if set(values)!={d['name'] for d in self.inputs}:raise ValueError('Inputs do not match context tensor names')
        inputs=[]
        for d in self.inputs:
            value=np.asarray(values[d['name']])
            if value.dtype!=np.dtype(d['dtype']) or value.shape!=d['shape']:raise ValueError('Wrong dtype/shape for '+d['name'])
            inputs.append(np.ascontiguousarray(value))
        outputs=[np.empty(d['shape'],dtype=d['dtype']) for d in self.outputs]
        ip=(C.c_void_p*len(inputs))(*(v.ctypes.data for v in inputs));op=(C.c_void_p*len(outputs))(*(v.ctypes.data for v in outputs))
        ib=(C.c_size_t*len(inputs))(*(v.nbytes for v in inputs));ob=(C.c_size_t*len(outputs))(*(v.nbytes for v in outputs))
        accelerator=C.c_double(-1);begin=time.perf_counter()
        rc=self.lib.nq_execute(self.handle,ip,ib,op,ob,C.byref(accelerator))
        elapsed=(time.perf_counter()-begin)*1000
        if rc:raise RuntimeError(f'QNN graphExecute returned {rc}')
        if accelerator.value<0:raise RuntimeError('HTP accelerator timing event missing')
        self.samples.append({'host_ms':elapsed,'accelerator_ms':accelerator.value/1000})
        return {d['name']:v for d,v in zip(self.outputs,outputs)}
    def close(self):
        if self.handle:self.lib.nq_close(self.handle);self.handle=None
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
