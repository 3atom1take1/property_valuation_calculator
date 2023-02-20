#!/usr/bin/env python
# cofing: utf-8

import pandas as pd
import re
import datetime
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

work_dir = os.getcwd()
print(work_dir+'/scraping_raw/suumo_baibai*.csv')
files = glob.glob(work_dir+'/scraping_raw/suumo_baibai*.csv')

input_file = files[0]