import xml.etree.ElementTree as ET
import glob, urllib, os, json, zlib, sys, getopt
import parse_xml as parse

movie_info = dict()
XML_PATH = "./en/xml/"
EN_TO_PATH = "./en/to/"
EN_FROM_PATH = "./en/from/"

target_files = ["bg-en.xml.gz", "cs-en.xml.gz", "da-en.xml.gz", "de-en.xml.gz", "el-en.xml.gz"]
source_files = ["en-es.xml.gz", "en-et.xml.gz", "en-fi.xml.gz", "en-fr.xml.gz", "en-he.xml.gz", "en-hr.xml.gz", "en-hu.xml.gz", "en-is.xml.gz", "en-it.xml.gz", "en-nl.xml.gz"]

def get_gz_files():
    get_target_files(source_files, EN_FROM_PATH)
    get_target_files(target_files, EN_TO_PATH)

def get_target_files(files, path):
    head = "http://opus.lingfil.uu.se/download.php?f=OpenSubtitles/xml/"
    folder = path
    for filename in files:
        urllib.urlretrieve(head + filename, folder + filename)
        print "Retrieved " + filename

def get_xml_files(filename, attr): # attr = 'toDoc' or 'fromDoc'
    tree = ET.parse(filename)
    root = tree.getroot()
    for link in root.iter("linkGrp"):
        name = link.attrib[attr]
        print "** Processing " + name
        url = "http://opus.lingfil.uu.se/download.php?f=OpenSubtitles/xml/" + name
        head, tail = os.path.split(name)
        new_filename = XML_PATH + tail
        save_genre_year(head, tail)
        urllib.urlretrieve(url, filename=new_filename)
        with open(new_filename, 'r') as gzip_file:
            decomp = zlib.decompressobj(16+zlib.MAX_WBITS)
            data = decomp.decompress(gzip_file.read())
            with open(new_filename[:len(new_filename)-3], 'w') as xml_file:
                xml_file.write(data)
        os.remove(new_filename)

def save_genre_year(path, filename):
    start = path.find('en/') + 3
    end = path[start:].find('/')
    genre = path[start:start + end]
    start = start + end + 1
    year = path[start:]
    movie_info[filename] = {'genre' : genre, 'year' : int(year)}

def get_all_xml_files(path, attrib):
    files = glob.glob(path + "*.xml")
    for filename in files:
        print "Processing " + filename
        get_xml_files(filename, attrib)

    with open('movie_info.json', 'w') as movie_file:
        movie_file.write(json.dumps(movie_info))

def extract_gz_files(path):
    compressed = glob.glob(path + "*.gz")
    for filename in compressed:
        print "Unzipping " + filename
        try:
            with open(filename, 'r') as f:
                decomp = zlib.decompressobj(16+zlib.MAX_WBITS)
                data = decomp.decompress(f.read())
                base_filename = os.path.basename(filename)
                base_filename = base_filename[:len(base_filename)-3]
                with open(path + base_filename, 'w') as xml_file:
                    xml_file.write(data)
                os.remove(filename)
        except Exception:                 
            print "Error with " + filename
            raise

def usage():
    print "Usage: python get_xml.py [-d] [-g] [-x]"
    print "-d = to download the gz files that have the names of xml files (step 1)"
    print "-g = to extract those gz files (step 2)"
    print "-x = to get the actual xml files (step 3)"
        
def main(argv):
    try:
        opts, args = getopt.getopt(argv, "dgx", ["download", "gz-extract", "xml-extract"])
        get_gz = extract_gz = extract_xml = False
        for opt, arg in opts:
            if opt in ('-d', '--download'):
                get_gz = True
            if opt in ('-g', '--gz-extract'):
                extract_gz = True
            if opt in ('-x', '--xml-extract'):
                extract_xml = True
        if get_gz:
            get_gz_files()
        if extract_gz:
            extract_gz_files(EN_FROM_PATH)
            extract_gz_files(EN_TO_PATH)
        if extract_xml:
            get_all_xml_files(EN_FROM_PATH, 'fromDoc')
            get_all_xml_files(EN_TO_PATH, 'toDoc')
                
    except getopt.GetoptError:
        usage()
        sys.exit(2)
        
if __name__ == "__main__":
    if len(sys.argv) == 1:
        usage()
        sys.exit(2)
    main(sys.argv[1:])
