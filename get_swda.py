import os, urllib, subprocess


URL = "http://compprag.christopherpotts.net/code-data/swda.zip"
DIR_NAME = "swda"
filename = os.path.basename(URL)
urllib.urlretrieve(URL, filename=filename)
subprocess.call(["unzip", filename])
os.remove(filename)
