# Stereo and Mono Depth Estimation Instructions

## Stereo Model

To run the stereo model, you will need:

- `requirements.txt`  
- The trained model  
- Input images (Left & Right)  
- Dataset folder  
- GPU  

Once you have everything ready, run the code as follows:

```bash
python test_psmnet.py \
    --loadmodel path/to/trained_model.tar \ 
    --left_folder path/to/left_images \
    --right_folder path/to/right_images \
    --model stackhourglass

```
## Notes

- You must run the stereo model code first to create the `predicted` folder.  
- Then you can run the evaluation code to assess the results.

## Mono Model

The mono model is fully set up for Google Colab, so no additional packages are required.

### Steps:

1. Run `depth_from_mono.ipynb` and save the output folder.  
2. Then run `D_Evaluation_Mono.ipynb` to evaluate the results.

