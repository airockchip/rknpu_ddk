#!/bin/bash

current_dir=`dirname $0`
cd $current_dir

doxygen

if [ $? == 0 ]
then
	echo "Document path: $PWD/rknpu_ddk_doc/"
fi
