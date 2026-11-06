"""Exact zero-based upstream validation cadence, strict comparison, no patience extension."""
import math
class NativeStop:
 def __init__(self):self.best=0.;self.best_epoch=None;self.stop_epoch=0;self.stopped=None
 def due(self,epoch0):return epoch0%3==0
 def observe(self,epoch0,forget,reserve):
  value=.5*(1-forget['mrr'])+.5*reserve['mrr']
  if not math.isfinite(value):raise FloatingPointError('Nonfinite development MRR_Avg')
  improved=value>self.best
  if improved:self.best=value;self.best_epoch=epoch0+1;self.stop_epoch=0
  else:self.stop_epoch+=1
  stop=self.stop_epoch>=1
  if stop:self.stopped=epoch0+1
  return dict(epoch_index=epoch0,epoch_one_based=epoch0+1,deleted_MRR=forget['mrr'],retained_MRR=reserve['mrr'],MRR_Avg=value,improved=improved,stop=stop)
