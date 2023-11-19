#!/bin/bash

#aws cloudformation create-stack --stack-name deepglobedata --template-body file:///Users/noeliaotero/Documents/WeCloudData/Capstone_project/SegmenSat/AWS_templates/cloudformation.yaml

aws cloudformation update-stack --stack-name deepglobedata --template-body file:///Users/noeliaotero/Documents/WeCloudData/Capstone_project/SegmenSat/AWS_templates/cloudformation.yaml

