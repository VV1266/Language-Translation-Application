import io
import simplejson
import json
import codecs


def load_config_file(filename):
    if filename is not None:
        try:
            with io.open(filename, encoding='utf-8') as f:
                file_config = simplejson.loads(f.read())
                return file_config
        except ValueError as e:
            print("Failed to read configuration file '{}'. Error: {}".format(filename, e))
        # self.override(file_config)
# functions for saving/opening objects
def jsonify(obj, out_file):
    """
	Inputs:
	- obj: the object to be jsonified
	- out_file: the file path where obj will be saved
	This function saves obj to the path out_file as a json file.
	"""
    json.dump(obj, codecs.open(out_file, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)


def read_txt_list(path):
    text_file = open(path, "r")
    return_list = []
    for line in text_file.readlines():
        return_list.append(line.strip().lower())
    return return_list
def dict_encode_utf_8(mydict):
    for (key, values) in mydict.items():
        my_string = u"{key}: {value}".format(key=key, value=values)
    return my_string