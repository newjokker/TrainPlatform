#!/bin/bash

# -----------------------------------------------------------------------------

UCD_CACHE_DIR="/usr/ucd_cache"
UCD_PATH="/usr/src/app/ucd"

# -----------------------------------------------------------------------------
ROOT_DIR=$1
LABEL_LIST=$2
TRAIN_UCD_NAME=$3
VAL_UCD_NAME=$4
#UCD_CACHE_DIR=$5
#UCD_PATH=$6
# -----------------------------------------------------------------------------

mkdir -p $UCD_CACHE_DIR
$UCD_PATH set cache_dir $UCD_CACHE_DIR
#
TRAIN_TXT_DIR=${ROOT_DIR}/txt_dir/train
VAL_TXT_DIR=${ROOT_DIR}/txt_dir/val
# 
TRAIN_UCD_PATH=${ROOT_DIR}/train_ucd.json
VAL_UCD_PATH=${ROOT_DIR}/val_ucd.json

# clean cache
#rm -r ${ROOT_DIR}
mkdir -p "${TRAIN_TXT_DIR}"
mkdir -p "${VAL_TXT_DIR}"


if [[ "$VAL_UCD_NAME" == "None" ]]; then
    echo "VAL_UCD_NAME is empty, get val by split ${TRAIN_UCD_PATH}"

  # load ucd
  echo "* load json ${TRAIN_UCD_NAME} -> ${TRAIN_UCD_PATH}"
  $UCD_PATH load "${TRAIN_UCD_NAME}"  "${TRAIN_UCD_PATH}"
  # split 
  echo split "${TRAIN_UCD_PATH}"
  $UCD_PATH split "${TRAIN_UCD_PATH}" "${TRAIN_UCD_PATH}" "${VAL_UCD_PATH}"  0.8

else

  # load ucd
  echo "* load json ${TRAIN_UCD_NAME} -> ${TRAIN_UCD_PATH}"
  $UCD_PATH load "${TRAIN_UCD_NAME}"  "${TRAIN_UCD_PATH}"

  echo "* load json ${VAL_UCD_NAME} -> ${VAL_UCD_PATH}"
  $UCD_PATH load "${VAL_UCD_NAME}"    "${VAL_UCD_PATH}"

fi


if [ ! -f "${TRAIN_UCD_PATH}" ]; then
  echo "* load ${VAL_UCD_NAME} -> ${TRAIN_UCD_PATH} failed "
  exit 100
fi

if [ ! -f "${VAL_UCD_PATH}" ]; then
  echo "* load ${VAL_UCD_NAME} -> ${VAL_UCD_PATH} failed"
  exit 100
fi

echo "* compare ${TRAIN_UCD_PATH} and ${VAL_UCD_PATH}"
$UCD_PATH diff "${TRAIN_UCD_PATH}"  "${VAL_UCD_PATH}"

# load img
echo "* load img ${TRAIN_UCD_PATH} -> ucd_cache"
$UCD_PATH save_cache "${TRAIN_UCD_PATH}" 10

echo "* load img ${VAL_UCD_PATH}  -> ucd_cache"
$UCD_PATH save_cache "${VAL_UCD_PATH}" 10

# to_yolo_txt
echo "* json to yolo txt -> ${TRAIN_TXT_DIR} -> ${LABEL_LIST}"
$UCD_PATH to_yolo_txt "${TRAIN_UCD_PATH}"  "${TRAIN_TXT_DIR}"  "${LABEL_LIST}"

echo "* json to yolo txt -> ${VAL_TXT_DIR} -> ${LABEL_LIST}"
$UCD_PATH to_yolo_txt "${VAL_UCD_PATH}"    "${VAL_TXT_DIR}"  "${LABEL_LIST}"

echo "* count_tags -> ${TRAIN_UCD_PATH}"
$UCD_PATH count_tags "${TRAIN_UCD_PATH}"

echo "* count_tags -> ${VAL_UCD_PATH}"
$UCD_PATH count_tags "${VAL_UCD_PATH}"



