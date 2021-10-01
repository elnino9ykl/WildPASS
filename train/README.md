**Example**
```
CUDA_VISIBLE_DEVICES=0,1,2,3
python3 segment.py 
--basedir /cvhci/data/
--num-epochs 200
--batch-size 12
--savedir /save_ecanet
--datasets 'MAP' 'IDD20K'
--num-samples 20000
--model ecanet
