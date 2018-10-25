# read in text files
novels = ['emma.txt','mansfieldpark.txt','northanger.txt','persuasion.txt','pridenp.txt','sensensense.txt']
inbooks = []
for novel in novels:
    f = open('./data/' + novel, encoding = 'utf-8')
    inbooks.append(f.read())

emma = inbooks[0]
mansfield = inbooks[1]
northanger = inbooks[2]
persuasion = inbooks[3]
pridenp = inbooks[4]
sensensense = inbooks[5]
# remove 30 lines, punctuation, chapter headers, later split into paragraphs
