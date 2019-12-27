<!--<h3><b>Data PrepareR</b></h3>-->
## <b>Ratio Image Based Rendering for Deep Single-Image Portrait Relighting</b> [[Project Page]](http://zhhoper.github.io/dpr.html) <br>
This is part of the Deep Portrait Relighting project. If you find this project useful, please cite the paper:
```
@InProceedings{DPR, 
  title={Deep Single Portrait Image Relighting},
  author = {Hao Zhou and Sunil Hadap and Kalyan Sunkavalli and David W. Jacobs},
  booktitle={International Conference on Computer Vision (ICCV)},
  year={2019}
}
```

### NOTE:
This code is not optimized and may not be well organized. 

### Dependences:
3DDFA: https://github.com/cleardusk/3DDFA (download the code and put it in useful_code, follow the instruction to download model and setup the code)

### Environment setup:
I use miniconda to setup virtual environment
* Create a virtual enviroment named RI_render (you can choose your own name): `conda create -n RI_render python=3.6`
* Install pytorch: `conda install  pytorch torchvision cudatoolkit=9.2 -c pytorch -n RI_render`
* Install dlib: `conda install  -c conda-forge  dlib -n RI_render`
* Install opencv: `conda install -n RI_render -c conda-forge opencv`
* Install scipy: `conda install -n RI_render -c conda-forge scipy`
* Install matplotlib: `conda install -n RI_render -c conda-forge matplotlib`
* Install cython: `conda install -n RI_render -c anaconda cython`
* Compile 3DDFA as mentioned in the github webpage
* Compile cython in utils/cython, follow the readme file
* Install Delaunay Triangulation:
    * Download Berkeley triangulation code from (http://www.cs.cmu.edu/~quake/triangle.html)
    * Unzip it into useful_code/triangle_berkeley/: unzip <path to the zip file> -d <path to the root>/useful_code/triangle_berkeley
* Install libigl: 
    * Download libig from https://github.com/libigl/libigl
    * Go to utils/libigl_arap, in `my_arap.sh` change the path of eigen and libigl
    * Run `bash my_arap.sh`
* Install shtools: https://github.com/SHTOOLS/SHTOOLS
* Install cvxpy: conda install -c conda-forge cvxpy

### Steps for rendering
1.  fitting 3DDFA: run bash `run_fit.sh`, will generate several files in result:
        *_3DDFA.png: draw 2D landmark on face
        *_depth.png: depth image
        *_detected.txt: detected 2D landmark on faces
        *_project.txt: projected 3D landmark
        *.obj: fitted mesh

2.  run bash `run_render.sh`
         generate albedo, normal, uv map and semantic segmentation:
         *_new.obj: obj file for rendering
         in render:
         *.png show generate images
         *.npy show original file of albedo, normal, uv map and semantic segmentation. NOTE: if you can install OpenEXR, you can save npy as .exr file

3.  run bash `run_node.sh`
         Apply arap to further align faces
         in render:
         generate arap.obj an object of arap algorithm
         *.node and *.ele temperal files for applying arap

         
4.  run `bash run_warp.sh`
         create warped albedo, normal, semantic segmentation in result/warp:
         
5.  run `bash run_fillHoles.sh`
         remove ear and neck region and fill in holes in generated normal map:
         create full_normal_faceRegion_faceBoundary_extend.npy and full_normal_faceRegion_faceBoundary_extend.png in result/warp

6.  run `bash run_relight.sh`
         relighting faces
         download our processed bip2017 lighting through (https://drive.google.com/open?id=1l0SiR10jBqACiOeAvsXSXAufUtZ-VhxC), change line 155 in script_relighting.py to poit to the lighting folder 
         Apply face semantic segmentation to get skin region of the face: https://github.com/Liusifei/Face_Parsing_2016 save the results in folder face_parsing/ (examples are shown in face_parsing, you can also skip this by adapting the code of script_relighting.py)
