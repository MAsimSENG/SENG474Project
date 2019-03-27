#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 18:16:19 2019

@author: Asim
"""
import os 
import random


def main():    
   all_data =  os.listdir("Dataset")
   
   
   test_data = random.sample(all_data,150 ) 
   
   trainData = [x for x in all_data if x not in test_data]
   
   
   for file in test_data:
       os.rename("Dataset/"+file, "Testset/"+file)
       


        
    
    
    
    
    


if __name__ == "__main__":
    main()