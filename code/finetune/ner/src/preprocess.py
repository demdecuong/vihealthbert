def write_txt(data, path):
    with open(path, 'w') as f:
        for item in data:
            f.write(item + '\n')
    return data


def read_conll(path):
    src = []
    trg = []
    labels = []
    with open(path, 'r') as f:
        tmp_src = []
        tmp_label = []
        for line in f:
            line = line.replace('\n', '').split(' ')
            if line[0] == '':
                src.append(' '.join(tmp_src))
                trg.append(' '.join(tmp_label))
                tmp_src = []
                tmp_label = []
            else:
                tmp_src.append(line[0])
                try:
                    tmp_label.append(line[1])
                except:
                    tmp_label.append('O')
                if len(line) >= 2 and line[1] not in labels:
                    labels.append(line[1])
    print(labels)
    return src, trg


test_src, test_trg = read_conll('../data/vinai_covid_word/test_word.conll')
dev_src, dev_trg = read_conll('../data/vinai_covid_word/dev_word.conll')
tr_src, tr_trg = read_conll('../data/vinai_covid_word/train_word.conll')

write_txt(test_src, '../data/vinai_covid_word/test/seq.in')
write_txt(test_trg, '../data/vinai_covid_word/test/seq.out')
write_txt(dev_src, '../data/vinai_covid_word/dev/seq.in')
write_txt(dev_trg, '../data/vinai_covid_word/dev/seq.out')
write_txt(tr_src, '../data/vinai_covid_word/train/seq.in')
write_txt(tr_trg, '../data/vinai_covid_word/train/seq.out')

a = ['O', 'B-DATE', 'I-DATE', 'B-NAME', 'B-AGE', 'B-LOCATION', 'I-LOCATION', 'B-JOB', 'I-JOB', 'B-ORGANIZATION', 'I-ORGANIZATION', 'B-PATIENT_ID',
    'B-SYMPTOM_AND_DISEASE', 'I-SYMPTOM_AND_DISEASE', 'B-GENDER', 'B-TRANSPORTATION', 'I-TRANSPORTATION', 'I-NAME', 'I-PATIENT_ID', 'I-AGE', 'I-GENDER']
write_txt(a, 'data/vinai_covid_word/slot_labels.txt')

