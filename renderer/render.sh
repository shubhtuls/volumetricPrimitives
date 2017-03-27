blenderExec=$1
blendFile=$2
objpath=$3
pngpath=$4
$blenderExec $blendFile --background --python ../renderer/renderBatch.py -- $objpath $pngpath > /dev/null