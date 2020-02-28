from nltk.corpus import stopwords
import nltk
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize, sent_tokenize
ps = PorterStemmer() 
import spacy
nlp = spacy.load('en_core_web_sm')
import neuralcoref
coref = neuralcoref.NeuralCoref(nlp.vocab) # initialize the neuralcoref with spacy's vocabulary
nlp.add_pipe(coref, name='neuralcoref') #add the coref model to pipe
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
from sklearn.metrics.pairwise import cosine_similarity
print("Model Loaded")
import tensorflow as tf
import json

def chunk(sent):
	words = word_tokenize(sent) 
	arr=[]
	   
	for w in words: 
	    arr.append(w) 
	li=[word for word in arr]
	#print(li)
	# stop_words = set(stopwords.words('english'))
	conj = set(('and', 'or' ,'but','while','so'))
	l=[1,2,3]
	tagged_list=[[]]
	i=0
	doc=nlp(sent)
	#print(sent)
	for token in doc:
		if (token.lemma_=="be"):
			l[0]="be"
		else:
			l[0]=token.text
		l[1]=token.tag_
		if(token.dep_=='nsubj'):
			l[2]=1
		else:
			l[2]=0
		tagged_list.insert(i,l)
		l=[1,2,3]
		i=i+1
	#print(tagged_list)
	#to find subjects in passive sentence
	noun=-1
	i=0
	while(i<len(tagged_list)-1):
		if(tagged_list[i][1].find("NN")!=-1):
			noun=i
		if(tagged_list[i][0]=='be'):
			if(i<len(tagged_list)-2 and tagged_list[i+1][1].startswith("V") and not tagged_list[i+1][1].startswith("VBG")):
				tagged_list[noun][2]=1
		i=i+1
	#print(tagged_list)


	n=[[]]
	ind=0
	ind2=-1
	i=0
	subj=""
	lis=[]
	flag=-1
	find=-1
	oind=-1
	while(i<len(tagged_list)-1):
		if(tagged_list[i][2]==1 and tagged_list[i][1]!="PRP" ):
			subj=tagged_list[i][0]
			oind=-1
		if(tagged_list[i][1]=="CC"  or tagged_list[i][0] in conj or tagged_list[i][0]=="," or tagged_list[i][0]==";" or tagged_list[i][0]=="."):
			j=i+1
			while(j<len(tagged_list)-1 and tagged_list[j][1].find("NN")==-1 and tagged_list[j][1].find("VB")==-1):
				j=j+1
			if(j<len(tagged_list)-1and tagged_list[j][1].find("NN")!=-1):
				if(tagged_list[i-1][2]==1 or tagged_list[i-1][2]==2):   #multiple subjects like plants and animals eat food
					ind2=-1
					tagged_list[j][2]=2
					subj=subj+" "+tagged_list[i][0]+" "+tagged_list[j][0]
				elif(tagged_list[j][2]==1):          #new subject ram eats and sita plays
					print('entered elif')
					n.append([tagged_list[x][0] for x in range(ind,i)])
					lis=[]
					ind =i+1

				else:                         # multiple objects ram eats plants and animals
					if(ind2==-1):
						ind2=ind 
						oind=i-1
					ct=ind2
					while(ct<i-1 and ct!=oind):
						ct=ct+1
					if(flag!=ind2):
						n.append([tagged_list[x][0] for x in range(ind2,i)])
						flag=ind2
						find=ct
					else:
						find=find+1					
						while(find<len(tagged_list)-1 and (tagged_list[find][1]!="CC"  and tagged_list[find][0] not in conj and tagged_list[find][0]!="," and tagged_list[find][0]!=";" and tagged_list[find][0]!=".")):
							find=find+1
						n.append([tagged_list[x][0] for x in range(ind2,i) if(x not in range(ct,find+1))])
					ind=i+1 #ADDED NOW



			elif(j<len(tagged_list)-1 and tagged_list[j][1].find("VB")!=-1):
				oind=-1
				if(i+1<len(tagged_list)-1 and tagged_list[i+1][1]!="PRP"):
					tagged_list[i][0]=subj
				#print(2,tagged_list[i][0],i)
				if(ind2!=-1 and ind2!=ind):
					find=find+1					
					while(find<len(tagged_list)-1 and (tagged_list[find][1]!="CC"  and tagged_list[find][0] not in conj and tagged_list[find][0]!="," and tagged_list[find][0]!=";" and tagged_list[find][0]!=".")):
						find=find+1
					n.append([tagged_list[x][0] for x in range(ind2,i) if(x not in range(ct,find+1))])
					ind2=-1
				else:
					for x in range(ind,i): #TO SEPARATE SUBJECTS
						if(tagged_list[x][1]=="CC"  or tagged_list[x][0] in conj or tagged_list[x][0]=="," or tagged_list[x][0]==";" or tagged_list[x][0]=="."):
							if(x>ind and x<i-1):
								if((tagged_list[x-1][2]== 1 or tagged_list[x-1][2]==2)):
									y=x+1
									while(y<len(tagged_list)-1 and tagged_list[y][1].find("NN")==-1 and tagged_list[y][1].find("VB")==-1):
										y=y+1
									if(tagged_list[y][2]==1 or tagged_list[y][2]==2):
										if(len(lis)==0):
											lis.append(x-1)
										lis.append(y)
					for l in range(len(lis)):
						n.append([tagged_list[x][0] for x in range(ind,i) if(x == lis[l] or x>lis[len(lis)-1]) or (l==0 and x<lis[0]) or (l>0 and x>lis[l-1]) and x<=lis[l]])
					if(len(lis)==0):
						n.append([tagged_list[x][0] for x in range(ind,i)])
				lis=[]
				ind=i;

		#print(subj)
		if(tagged_list[i][1]=="PRP"):
			tagged_list[i][0]=subj
		i=i+1
	if(ind2!=-1 and ind2!=ind):
		#print("ind= ",ind,"ind2=",ind2," i=",i)
		find=find+1					
		while(find<len(tagged_list)-1 and (tagged_list[find][1]!="CC"  and tagged_list[find][0] not in conj and tagged_list[find][0]!="," and tagged_list[find][0]!=";" and tagged_list[find][0]!=".")):
			find=find+1
		n.append([tagged_list[x][0] for x in range(ind2,i) if(x not in range(ct,find+1))])
		ind2=-1;
	else:	
		for x in range(ind,i):
			if(tagged_list[x][1]=="CC"  or tagged_list[x][0] in conj or tagged_list[x][0]=="," or tagged_list[x][0]==";" or tagged_list[x][0]=="."):
				if(x>ind and x<i-1):
					if((tagged_list[x-1][2]== 1 or tagged_list[x-1][2]==2)):
						y=x+1
						while(y<len(tagged_list)-1 and tagged_list[y][1].find("NN")==-1 and tagged_list[y][1].find("VB")==-1):
							y=y+1
						if(tagged_list[y][2]==1 or tagged_list[y][2]==2):
							if(len(lis)==0):
								lis.append(x-1)
							lis.append(y)
		for l in range(len(lis)):
			n.append([tagged_list[x][0] for x in range(ind,i) if(x == lis[l] or x>lis[len(lis)-1]) or (l==0 and x<lis[0]) or (l>0 and x>lis[l-1]) and x<=lis[l]])
		if(len(lis)==0):
			n.append([tagged_list[x][0] for x in range(ind,i)])
	lis=[]
	#print(tagged_list)
	#print()
	return n

def res_coref(P):
    doc=nlp(P)
    return doc._.coref_resolved

def cos_sim(A,B):
    res = cosine_similarity([A],[B])
    print(res)
    return res[0]

def find_sentences(X,Y):
    '''
        input:
            X - The right answer
            Y - The student's answer
        output:
            returns lists L1 and L2 as described below
    '''
    L1 = [ ] # list of sentences that are good - i.e are found in both right and student answer
    L2 = [ ] # list of sentences that are bad - i.e not found in student answer but present in right answer
    threshold = 0.80000 #the threshold to be used to accepting sentences
    X_orig = sent_tokenize(X) 
    Y_orig = sent_tokenize(Y)
    X = res_coref(X)
    Y = res_coref(Y)
    X = sent_tokenize(X)
    Y = sent_tokenize(Y)
    X_enc = model.encode(X)
    Y_enc = model.encode(Y)
    X_len = len(X)
    Y_len = len(Y)
    for i in range(0,X_len):
        flg=0
        for j in range(0,Y_len):
            if cos_sim(X_enc[i],Y_enc[j])>= threshold:
                L1.append((X_orig[i],Y_orig[j]))
                flg=1
                break
        if flg==0:
            L2.append(X_orig[i])
    return L1,L2

def print_formatted(X,Y):
    L1,L2=find_sentences(X,Y)
    for i in L1:
        print("Sentence in Right answer: "+i[0]+'\n'+"Sentence in student answer:"+i[1])
    print()
    for i in L2:
        print("Sentence in Right answer but missing from student's answer: "+i)

def compare(right_answer,student_answer):
	l=chunk(right_answer)
	for i in range(1,len(l)):
		l[i]=" ".join(l[i])
	l=l[1:]
	l=". ".join(l)
	print_formatted(l,student_answer)


student_answer = ['Plants take in water. Plants give out oxygen']








def read_squad_examples(input_file,student_answer):
  """Read a SQuAD json file into a list of SquadExample."""
  l=[]
  ct=0
  with tf.gfile.Open(input_file, "r") as reader:
    input_data = json.load(reader)
    for i in input_data:
    	compare(input_data[i],student_answer[ct])
    	ct=ct+1

read_squad_examples('output2/predictions.json',student_answer)

