import os
if __name__ == '__main__':
    file = 'lexicons/subjectivity_lexicon/subjclueslen1-HLTEMNLP05.tff'
    with open ('lexicons/processed_subj.txt', 'w') as outFile:
        with open(file) as input:
            lines = input.readlines()
            for line in lines:
                fields = line.split()
                if fields[0].split('=')[1] == 'strongsubj':
                    word = fields[2].split('=')[1]
                    pos = fields[3].split('=')[1]
                    sentiment = fields[5].split('=')[1]
                    outFile.write(word + ' ' + pos + ' ' + sentiment + '\n')