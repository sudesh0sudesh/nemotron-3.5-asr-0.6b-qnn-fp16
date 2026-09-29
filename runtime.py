# SPDX-License-Identifier: Apache-2.0
"""HTP-only neural runtime, extracted from the Arduino-validated implementation.

Frontend/cache reference: soniqo/speech-core commit
f0050757a7d7d24c60b17e60016989ecd07a491d (Apache-2.0).
Modified for native QNN HTP execution and NPU-side token selection; see NOTICE.md.
"""
import re
import time
import numpy as np
from nemotron_qnn_context import QnnContext

class NativeEncoder:
    def __init__(self,binary):
        from nemotron_qnn_context import QnnContext
        self.context=QnnContext(binary)
        self.input_names=['audio_signal','audio_length','language_mask','pre_cache','cache_last_channel','cache_last_time','cache_last_channel_len']
        self.output_names=['encoded_output','encoded_length','new_pre_cache','new_cache_last_channel','new_cache_last_time','new_cache_last_channel_len']
        assert {d['name'] for d in self.context.inputs}==set(self.input_names)
        available={d['name'] for d in self.context.outputs}
        self.output_map={name:name if name in available else f'output_{i}' for i,name in enumerate(self.output_names)}
        assert set(self.output_map.values())==available
        expected_shapes=[(1,4,1024),(1,),(1,128,9),(24,1,56,1024),(24,1,1024,8),(1,)]
        specifications={d['name']:d for d in self.context.outputs}
        assert all(specifications[self.output_map[name]]['shape']==shape for name,shape in zip(self.output_names,expected_shapes))
        assert all(d['dtype'] in ['float32','int32'] for d in self.context.inputs+self.context.outputs), 'Expected explicit float32/int32 graph IO'
        self.spec={d['name']:d for d in self.context.inputs}
        self.total_ms=0.;self.calls=0;self.delegate_count=1
    def roles(self,values):
        if not 0<=int(values[6][0])<=56:raise ValueError('Encoder cache length is outside its compiled 0..56 contract')
        inputs={name:np.asarray(values[i],dtype=self.spec[name]['dtype']) for i,name in enumerate(self.input_names)}
        result=self.context.run(inputs)
        self.calls+=1;self.total_ms+=self.context.samples[-1]['host_ms']
        return {i:result[self.output_map[name]] for i,name in enumerate(self.output_names)}

def mel_features(audio):
    # Match the published LiteRT streaming frontend: preemphasis, left reflect
    # padding, symmetric 400-sample Hann, 512-point FFT, Slaney-normalized mel.
    chunks=(len(audio)+5119)//5120
    audio=np.pad(audio,(0,chunks*5120-len(audio))).astype(np.float32)
    emphasized=audio.copy();emphasized[1:]=audio[1:]-np.float32(.97)*audio[:-1]
    emphasized=np.pad(emphasized,(256,0),mode='reflect')
    frames=np.lib.stride_tricks.sliding_window_view(emphasized,400)[::160]
    window=np.hanning(400).astype(np.float32)
    power=np.abs(np.fft.rfft(frames*window,n=512,axis=-1))**2
    log_step=27/np.log(6.4)
    high_mel=15+np.log(8)*log_step
    mels=np.linspace(0,high_mel,130)
    hz=np.where(mels<15,mels*200/3,1000*np.exp((mels-15)/log_step))
    bins=np.arange(257)*16000/512
    bank=np.maximum(0,np.minimum((bins[None,:]-hz[:-2,None])/(hz[1:-1,None]-hz[:-2,None]),(hz[2:,None]-bins[None,:])/(hz[2:,None]-hz[1:-1,None])))
    bank*=2/(hz[2:,None]-hz[:-2,None])
    mel=np.log(bank@power.T+2**-24).astype(np.float32)
    assert mel.shape==(128,chunks*32),mel.shape
    return mel

class NativeGraph:
    def __init__(self,binary,inputs,outputs):
        self.context=QnnContext(binary)
        self.input_names=inputs;self.output_names=outputs
        try:
            assert {d['name'] for d in self.context.inputs}==set(inputs)
            available={d['name'] for d in self.context.outputs}
            self.output_map={name:name if name in available else f'output_{i}' for i,name in enumerate(outputs)}
            assert set(self.output_map.values())==available
            self.spec={d['name']:d for d in self.context.inputs}
        except Exception:
            self.context.close();raise
    def run(self,values):
        result=self.context.run({name:np.asarray(values[name],dtype=self.spec[name]['dtype']) for name in self.input_names})
        if not all(np.isfinite(v).all() for v in result.values()):raise RuntimeError('Nonfinite HTP output')
        return {name:result[self.output_map[name]] for name in self.output_names}

def transcribe(encoder,decoder,joint,vocab,language,audio):
    graphs={'encoder':encoder,'decoder':decoder,'joint':joint}
    before={k:len(g.context.samples) for k,g in graphs.items()}
    started=time.perf_counter();mel=mel_features(audio)
    mask=np.zeros((1,128),np.float32);mask[0,int(language)]=1
    state={2:np.zeros((1,128,9),np.float32),3:np.zeros((24,1,56,1024),np.float32),4:np.zeros((24,1,1024,8),np.float32),5:np.zeros((1,),np.int32)}
    h=np.zeros((2,1,640),np.float32);c=h.copy();token=13087;tokens=[];prediction=None;per_chunk=[];limit_hits=0
    for offset in range(0,mel.shape[1],32):
        begin=time.perf_counter()
        result=encoder.roles({0:mel[:,offset:offset+32][None],1:np.array([32],np.int32),2:mask,3:state[2],4:state[3],5:state[4],6:state[5]})
        state={k:result[k] for k in [2,3,4,5]}
        if not np.isfinite(result[0]).all():raise RuntimeError('Nonfinite encoder features')
        for frame in result[0][0,:int(result[1][0])]:
            for _ in range(10):
                if prediction is None:prediction=decoder.run({'token':np.array([[token]],np.int32),'h':h,'c':c})
                best=int(joint.run({'encoder_output':frame.reshape(1,1,1024),'decoder_output':prediction['decoder_output']})['token_id'].item())
                if not 0<=best<=13087:raise RuntimeError('Invalid NPU token')
                if best==13087:break
                tokens.append(best);token=best;h=prediction['h_out'];c=prediction['c_out'];prediction=None
            else:limit_hits+=1
        per_chunk.append((time.perf_counter()-begin)*1000)
    elapsed=(time.perf_counter()-started)*1000
    text=''.join(vocab[str(t)] for t in tokens).replace('▁',' ').strip()
    text=re.sub(r'<[a-z]{2,3}-[A-Z]{2}>','',text).strip()
    samples={k:g.context.samples[before[k]:] for k,g in graphs.items()}
    return {'text':text,'tokens':tokens,'processing_ms':elapsed,'chunk_ms':per_chunk,'symbol_limit_hits':limit_hits,
            'qnn_samples':samples,'graph_timings':{k:{'calls':len(s),'host_ms':sum(x['host_ms'] for x in s),'accelerator_ms':sum(x['accelerator_ms'] for x in s)} for k,s in samples.items()}}
