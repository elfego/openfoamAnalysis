#!/bin/bash

OFENV=$1

echo "Copying to ${OFENV}/src/functionObjects/field"
cp -r R ${OFENV}/src/functionObjects/field

echo "Adding to the Makefile list"
grep 'R/R.C' ${OFENV}/src/functionObjects/field/Make/files && exit
sed --in-place=orig 's/Q\/Q.C/Q\/Q.C\nR\/R.C/g' ${OFENV}/src/functionObjects/field/Make/files

echo "Copying to ${OFENV}/etc/caseDicts/postProcessing/fields/R"
ls ${OFENV}/etc/caseDicts/postProcessing/fields/R.orig || \
    cp ${OFENV}/etc/caseDicts/postProcessing/fields/R ${OFENV}/etc/caseDicts/postProcessing/fields/R.orig
cp etcR ${OFENV}/etc/caseDicts/postProcessing/fields/R
