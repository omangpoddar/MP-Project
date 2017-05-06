from pymining import seqmining

def isSubSequence(string1, string2, m, n):
    # Base Cases
    if m == 0:    return True
    if n == 0:    return False
 
    # If last characters of two strings are matching
    if string1[m-1] == string2[n-1]:
        return isSubSequence(string1, string2, m-1, n-1)
 
    # If last characters are not matching
    return isSubSequence(string1, string2, m, n-1)

seqs = ( 'caabc', 'abcb', 'cabc', 'abbca')
freq_seqs = seqmining.freq_seq_enum(seqs, 2)
a=list(freq_seqs)
a1=[]
print(a)
for i in range(len(freq_seqs)):
        s1=""
        for j in range(len(a[i][0])):
                s1 = s1+(a[i][0][j])
        a1.append(s1)
print(a1)
file = open("t.txt","w")
for i in range(len(a1)):
        file.write(a1[i]+"\n")
file.close() 
string2 = "abc"
n = len(string2)
a2=[]
for i in range(len(a1)):
        m = len(a1[i])
        string1=a1[i]
        if isSubSequence(string1, string2, m, n):
                a2.append(1)
        else:
                a2.append(0)
print(a2)      
#for item in freq_seqs: 
#        print(item)
#print(len(freq_seqs.pop()[0]))