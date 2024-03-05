import os

def walk_directory(directory, formatList=[]):
    filePaths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.strip().split('.')[-1] not in formatList:
                continue
            filePath = os.path.join(root, file)
            # print(filePath)
            filePaths.append(filePath)
    return filePaths

def main():
    pass

if __name__ == '__main__':
    main()
