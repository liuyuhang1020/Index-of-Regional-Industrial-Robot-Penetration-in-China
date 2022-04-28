import pandas

def standardize(sentence,root,phrase):
    j1=jushi1(sentence,root,phrase)
    if j1:
        return j1
    j2=jushi2(sentence,root,phrase)
    if j2:
        return j2
    j3=jushi3(sentence,root,phrase)
    if j3:
        return j3
    j4=jushi4(sentence,root,phrase)
    if j4:
        return j4
    return 0

def jushi1(sentence,root,phrase):
    if "的" not in sentence:
        return 0
    else:
        verbroot_returnlist=[]
        verbphrase_returnlist=[]
        verbroot=list(root["verb"])
        verbphrase=list(phrase["verb"])
        verb=(sentence.split("的")[1]).split("、")
        for v in verb:
            if v in verbroot:
                verbroot_returnlist.append(v)
            elif v in verbphrase:
                verbphrase_returnlist.append(v)          
            else:
                return 0
    noun=(sentence.split("的")[0]).split("、")
    standard=[]
    for n in noun:
        if (n in verbroot) or (n in verbphrase):
            return 0
        else:
            for v in verbroot_returnlist:
                standard.append(n+v)
            for v in verbphrase_returnlist:
                head=list(phrase)
                head.remove("verb")
                for i in head:
                    if not list(pandas.isnull(phrase[phrase["verb"]==v][i]))[0]:
                        standard.append(n+list(phrase[phrase["verb"]==v][i])[0])
    return standard

def jushi2(sentence,root,phrase):
    if "：" not in sentence:
        return 0
    else:
        verbroot_returnlist=[]
        verbphrase_returnlist=[]
        verbroot=list(root["verb"])
        verbphrase=list(phrase["verb"])
        verb=(sentence.split("：")[0]).split("、")
        for v in verb:
            if v in verbroot:
                verbroot_returnlist.append(v)
            elif v in verbphrase:
                verbphrase_returnlist.append(v)
            else:
                return 0
    noun=(sentence.split("：")[1]).split("、")
    standard=[]
    for n in noun:
        if (n in verbroot) or (n in verbphrase):
            return 0
        else:
            for v in verbroot_returnlist:
                standard.append(n+v)
            for v in verbphrase_returnlist:
                head=list(phrase)
                head.remove("verb")
                for i in head:
                    if not list(pandas.isnull(phrase[phrase["verb"]==v][i]))[0]:
                        standard.append(n+list(phrase[phrase["verb"]==v][i])[0])
    return standard

def jushi3(sentence,root,phrase):
    if "的" in sentence or "：" in sentence:
        return 0
    else:
        noun=[]
        standard=[]
        verbroot_returnlist=[]
        verbphrase_returnlist=[]
        verbroot=list(root["verb"])
        verbphrase=list(phrase["verb"])
        word=sentence.split("、")
        w=len(word)-1
        t=-2
        while w!=t and w>=0:
            t=w
            if word[w] in verbroot:
                verbroot_returnlist.append(word[w])
                w-=1
            elif word[w] in verbphrase:
                verbphrase_returnlist.append(word[w])
                w-=1
        if w<0:
            return 0
        endswithverb=0
        for v in verbphrase:
            if word[w].endswith(v):
                verbphrase_returnlist.append(v)
                noun.append(word[w].replace(v,""))
                endswithverb=1
                break            
        if not endswithverb:
            for v in verbroot:
                if word[w].endswith(v):
                    verbroot_returnlist.append(v)
                    noun.append(word[w].replace(v,""))
                    endswithverb=1
                    break
        if not endswithverb:
            return 0
        w-=1
        while w>=0:
                noun.append(word[w])
                w-=1
        for n in noun:
            if (n in verbroot) or (n in verbphrase):
                return 0
            else:
                for v in verbroot_returnlist:
                    standard.append(n+v)
                for v in verbphrase_returnlist:
                    head=list(phrase)
                    head.remove("verb")
                    for i in head:
                        if not list(pandas.isnull(phrase[phrase["verb"]==v][i]))[0]:
                            standard.append(n+list(phrase[phrase["verb"]==v][i])[0])
        return standard
    
def jushi4(sentence,root,phrase):
    if "的" in sentence or "：" in sentence:
        return 0
    else:
        noun=[]
        standard=[]
        verbroot_returnlist=[]
        verbphrase_returnlist=[]
        verbroot=list(root["verb"])
        verbphrase=list(phrase["verb"])
        word=sentence.split("、")
        w=0
        t=-2
        while w!=t and w<=len(word)-1:
            t=w
            if word[w] in verbroot:
                verbroot_returnlist.append(word[w])
                w+=1
            elif word[w] in verbphrase:
                verbphrase_returnlist.append(word[w])
                w+=1
        if w>len(word)-1:
            return 0
        startswithverb=0
        for v in verbphrase:
            if word[w].startswith(v):
                verbphrase_returnlist.append(v)
                noun.append(word[w].replace(v,""))
                startswithverb=1
                break 
        if not startswithverb:
            for v in verbroot:
                if word[w].startswith(v):
                    verbroot_returnlist.append(v)
                    noun.append(word[w].replace(v,""))
                    startswithverb=1
                    break
        if not startswithverb:
            return 0
        w+=1
        while w<=len(word)-1:
            noun.append(word[w])
            w+=1
        for n in noun:
            if (n in verbroot) or (n in verbphrase):
                return 0
            else:
                for v in verbroot_returnlist:
                    standard.append(n+v)
                for v in verbphrase_returnlist:
                    head=list(phrase)
                    head.remove("verb")
                    for i in head:
                        if not list(pandas.isnull(phrase[phrase["verb"]==v][i]))[0]:
                            standard.append(n+list(phrase[phrase["verb"]==v][i])[0])
        return standard