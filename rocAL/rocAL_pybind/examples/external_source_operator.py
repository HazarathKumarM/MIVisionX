import amd.rocal.fn as fn

pythonScript = ("def read_array_from_file(filename):\n"
               "   try:\n"
               "       with open(filename, 'r') as file:\n"
               "           data = [int(line.strip()) for line in file]\n"
               "       return data\n"
               "   except Exception as e:\n"
               "       print('Error reading file:', str(e))\n"
               "       return []\n")

fn.ExternalSource(source = pythonScript)