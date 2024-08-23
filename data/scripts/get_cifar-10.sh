#!/bin/bash

#Download/uzip images and labels
d="../datasets" #uzip directory
mkdir -p $d && cd $d

#f='cifar-10'
f='titanic'
prix='.zip'
#echo 'downloading' $url$f '...'
kaggle competitions download -c $f
unzip -q $f -d $d && rm $f$prix &