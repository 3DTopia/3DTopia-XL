CURRENT=$(pwd)
cd dva/mvp/extensions/mvpraymarch
make -j4
cd ../utils
make -j4
cd ${CURRENT}