import json
import re
import os
extractedData = {}
pattern = r"tensor\((-?\d+\.\d+)"

# with open(os.getcwd() + "/results.txt", 'r') as fp:
with open(os.getcwd() + "/results_self.txt", 'r') as fp:
    lines = fp.readlines()
    for eachLine in lines:

        if eachLine == '' or eachLine=='\n':
            continue
        contents = eachLine.strip().split(':')

        key = contents[0].strip() if contents[0] else 'None'
        val = contents[1].strip() if contents[1] else 0
        print(key, ':', val)
        if key == 'working on':
            fileName = val.split('/')[-1]
            extractedData[fileName] = {}
        elif key == 'similarity between Anchor and Positive(Target)':
            match = re.search(pattern, val)
            if match:
                val = eval(match.group(1))
            extractedData[fileName]['Similarity to Target'] = val
            similarity_to_target = val
        elif key == 'similarity between Anchor and Negative(Origin)':
            match = re.search(pattern, val)
            if match:
                val = eval(match.group(1))
            extractedData[fileName]['Similarity to Origin'] = val
            similarity_to_origin = val
            extractedData[fileName]['DS'] = True if similarity_to_origin <= 0.2 or similarity_to_target >= 0.8 else False
        elif key == 'stoi':
            extractedData[fileName]['STOI'] = val
        print(extractedData[fileName])
    print(len(extractedData))
# with open(os.getcwd() + '/logAnalyse/output.json', 'w') as f:
with open(os.getcwd() + '/logAnalyse/self-output.json', 'w') as f:
    json.dump(extractedData, f)

extractedData = {}
cnt = 0
with open(os.getcwd() + "/WAD_results_self.txt", 'r') as fp:
# with open(os.getcwd() + "/WAD_results.txt", 'r') as fp:
    lines = fp.readlines()
    for eachLine in lines:

        if eachLine == '' or eachLine=='\n':
            continue
        contents = eachLine.strip().split(':')

        key = contents[0].strip() if contents[0] else 'None'
        val = contents[1].strip() if contents[1] else 0

        if key == 'Original audio':
            print(val)
            fileName = val
            extractedData[fileName] = {}
        elif key == 'WAD':
            print(val)
            extractedData[fileName]['WAD'] = val
            cnt += 1
    print(len(extractedData))

# with open(os.getcwd() + "/logAnalyse/WADoutput.json", 'w') as f:
with open(os.getcwd() + "/logAnalyse/self-WADoutput.json", 'w') as f:
    json.dump(extractedData, f)
print(cnt)