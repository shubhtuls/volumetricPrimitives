blender=$1
objpath=$2
pngpath=$3
/home/eecs/shubhtuls/Downloads/blender-2.76/blender $blender --background --python ../renderer/renderBatch.py -- $objpath $pngpath > /dev/null
