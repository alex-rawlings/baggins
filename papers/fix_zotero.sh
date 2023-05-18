#!/bin/bash
sed -i .bak "s/\\\\textbackslash /\\\\/g" $1
sed -i .bak "s/\\\\aa p/\\\\aap/g" $1
rm $1.bak