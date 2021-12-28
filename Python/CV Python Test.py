b = 'potato'

def myfunc(a):
    myword = []
    for i in range(len(a)):
        if i % 2 == 0:
            myword.append(a[i].upper())
        else:
            myword.append(a[i].lower())
    return ''.join(myword)

print(myfunc(b))
