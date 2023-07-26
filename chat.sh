#!/bin/bash

MODEL="out/model.bin"

#------

if [ $# != 2 ]
then
  echo "Usage:`basename "$0"` state_file user_input"
  exit 0
fi

file=$1
input=$2

cd `dirname "$0"`

LF=$'\n'

prompt=`cat "$file"`
last_message=${prompt##*$LF}

ai_next="Lily replied to Timmy, \""
if [ "$last_message" != "$ai_next" ]
then
  prompt="${prompt}${LF}Timmy replied to Lily, \"${input}l\"${LF}${ai_next}"
fi

tmp_file=`mktemp`
./run "$MODEL" 0.0 65536 "$prompt" "\"" save  | tee "$tmp_file"
cp "$tmp_file" "$file"
