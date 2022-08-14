#!/usr/bin/env bash
unzip FGT_data.zip
unzip weights.zip
mkdir FGT/checkpoint
mkdir FGT/flowCheckPoint
mkdir LAFC/checkpoint
mv weights/fgt/* FGT/checkpoint
mv weights/lafc/* LAFC/checkpoint
mv weights/lafc_single/* FGT/flowCheckPoint
rm -r weights
