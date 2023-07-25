#!/bin/bash

cd `dirname "$0"`

input=`cat -`
file=$1

echo "Timmy replied to Lily, \"$input\"" >> "$file"
echo "Lily replied to Timmy, \"" >> "$file"
