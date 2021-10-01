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
```

**Steps** 
1. Download datasets for training (Mapillary Vistas, IDD20K used in our work, other datasets like Cityscapes and BDD can be also considered).

2. Create pseudo labels for omni-supervision, using the open PASS pipeline with data from the WildPASS2K set (this step is not necessary for running the code, but critical for the panoramic segmentation performance).
