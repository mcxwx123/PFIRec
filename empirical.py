import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances
from sklearn.metrics import cohen_kappa_score
import spacy
from textblob import TextBlob
import copy

def get_userdata(closer,i):
    for n in range(len(dfall)):
        if (dfall.at[n,'m']==closer)&(dfall.at[n,'i']==i):
            return dfall.at[n,'issuetitles'],dfall.at[n,'issuebodys'],dfall.at[n,'userallcmt'],dfall.at[n,'onemonth_cmt']

def get_solvet(issuet,clst):
    days=(clst-issuet)/86400000
    return days

def cos_sim(emb0,emb1):
    try:
        a=cosine_similarity(emb0,emb1)[0][0]
    except:
        return None
    return a

def euc_sim(emb0,emb1):
    try:
        a=1/(1+euclidean_distances(emb0, emb1)[0][0])
    except:
        return None
    return a

def lemma(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

def jaccard_sim(t0,t1):
    try:
        t0=lemma(t0)
        t1=lemma(t1)
        a = set(t0.split()) 
        b = set(t1.split())
        c = a.intersection(b)
    except:
        return None
    return float(len(c)) / (len(a) + len(b) - len(c))

def getsen(s):
    blob = TextBlob(s)
    return blob.sentiment.polarity

def cls_issue_sen_mean(titles,bodys):
    lst=[]
    for i in range(len(titles)):
        text=titles[i]+' '+bodys[i]
        lst.append([getsen(text)])
    if lst==[]:
        return 0
    else:
        return np.mean(lst)

def sort_lsts(lst):
    maxinds=[]
    copylst=copy.deepcopy(lst)
    for _ in range(len(lst[0])):
        indexes=list(range(len(lst[0])))
        for lstind in range(len(lst)):
            alst=[lst[lstind][m] for m in indexes]
            indinds=[i for i, x in enumerate(alst) if x == max(alst)]
            indexes = [indexes[m] for m in indinds]
            if len(indexes)==1 or lstind==len(lst)-1:
                maxinds.append(indexes[0])
                lst[0][indexes[0]]=-1
                break
    rlst=[[sublist[mi] for mi in maxinds] for sublist in copylst]
    return rlst

if __name__ == "__main__":
    nlp=spacy.load('en_core_web_sm')

    language=[["TypeScript","JavaScript","TypeScript","JavaScript","TypeScript"],
        ["Python","Python","Python","Python"],
        ["Python","Python","Python","Python"],
        ["JavaScript","JavaScript","C#","TypeScript"],
        ["Python","Python","Python","C++"],
        ["JavaScript","JavaScript","TypeScript","Python"],
        ["Python","Python","Python","Python"],
        ["JavaScript","JavaScript","TypeScript","TypeScript"],
        ["JavaScript","Python","TypeScript","Python"],
        ["Python","Python","Python","Python"],
        ["JavaScript","JavaScript","TypeScript","TypeScript"],
        ["TypeScript","Python","Python"],
        ["JavaScript","TypeScript","JavaScript"],
        ["JavaScript","TypeScript","TypeScript"],
        ["Python","Python","Python"],
        ["JavaScript","JavaScript","JavaScript"],
        ["Python","Python","Python"],
        ["JavaScript","JavaScript","JavaScript"],
        ["Python","Python","Python"],
        ["Python","TypeScript","Python"],
        ["TypeScript","C#","TypeScript"],
        ["Python","Python","Python"],
        ["Python","JavaScript","TypeScript"],
        ["JavaScript","JavaScript","TypeScript"],
        ["Python","TypeScript","Python"],
        ["JavaScript","JavaScript","JavaScript"],
        ["Python","Python","Python"],
        ["JavaScript","Python","C++"],
        ["JavaScript","Python","Python"],
        ["JavaScript","JavaScript","TypeScript"],
        ["TypeScript","JavaScript","TypeScript"],
        ["Python","TypeScript","Python"],
        ["JavaScript","JavaScript","JavaScript"],
        ["JavaScript","TypeScript","TypeScript"],
        ["Python","Python","Python"],
        ["JavaScript","TypeScript","TypeScript"],
        ["Java","PHP","PHP"]]
        
    tasktype=[[3,3,3,1,3],
        [1,1,1,1],
        [4,6,5,6],
        [4,4,1,1],
        [6,6,6,6],
        [1,4,1,1],
        [4,4,4,6],
        [1,2,1,3],
        [2,4,4,4],
        [4,4,4,4],
        [6,6,3,4],

        [6,6,6],
        [4,4,1],
        [3,1,3],
        [1,1,1],
        [1,1,1],
        [1,4,4],
        [6,1,1],
        [6,6,6],
        [1,1,1],
        [1,1,1],
        [4,1,4],
        [6,3,3],
        [1,3,4],
        [6,1,6],
        [1,1,1],
        [6,1,1],
        [3,3,1],
        [2,6,1],
        [1,6,6],
        [1,1,4],
        [3,6,1],
        [6,1,1],
        [1,4,3],
        [6,6,6],
        [1,1,1],
        [6,3,3]
        ]
    
    domain=[[4,4,4,5,1],
        [4,4,4,4],
        [4,4,4,4],
        [3,3,4,1],
        [4,4,4,1],
        [3,1,3,1],
        [4,4,4,4],
        [1,1,3,1],
        [1,4,3,4],
        [4,5,4,2],
        [3,3,1,1],
        [1,4,4],
        [5,4,4],
        [1,1,4],
        [4,5,4],
        [3,3,4],
        [1,4,4],
        [3,3,1],
        [5,4,4],
        [4,1,4],
        [4,5,1],
        [4,4,4],
        [4,5,1],
        [3,3,4],
        [4,1,4],
        [3,1,5],
        [4,4,4],
        [3,1,1],
        [1,4,4],
        [3,3,1],
        [4,4,1],
        [4,1,1],
        [5,5,4],
        [3,3,1],
        [5,4,4],
        [3,1,1],
        [1,1,1]
        ]

    tasktype1=[3,3,3,1,3,1,1,1,1,4,6,5,6,4,4,1,1,6,6,6,6,1,4,1,1,4,4,4,6,1,2,1,3,2,4,4,4,4,4,4,4,6,6,3,4,6,6,6,4,4,1,3,1,3,1,1,1,1,1,1,1,4,4,6,1,1,6,6,6,1,1,1,1,1,1,4,4,4,6,3,1,1,3,4,6,1,6,1,1,1,6,1,1,3,3,1,3,6,1,1,6,6,1,1,4,3,6,1,6,1,1,1,4,3,6,6,6,1,1,1,6,3,3]
    tasktype2=[3,3,3,1,3,1,1,1,1,4,6,4,6,4,4,1,1,6,6,6,6,1,4,1,1,4,4,4,6,1,2,1,3,2,4,4,4,4,4,4,4,6,6,3,4,6,4,6,4,6,1,3,1,3,1,1,1,1,1,1,1,4,4,6,1,1,6,6,6,1,1,1,1,1,1,4,1,4,6,3,3,1,3,4,6,1,6,1,1,1,6,1,1,3,3,1,2,6,1,1,6,6,1,1,4,3,6,1,6,1,1,1,4,3,6,6,6,1,1,1,6,3,3]
    domain1=[4,4,4,5,1,4,4,4,4,4,4,4,4,3,3,4,6,4,4,4,1,3,1,3,1,4,4,4,4,1,1,3,1,1,4,3,4,4,5,4,2,3,3,1,6,1,4,4,5,4,4,1,1,4,4,5,4,3,3,4,1,4,4,3,3,1,5,4,4,4,1,4,4,5,1,4,4,4,4,5,1,3,3,4,4,6,4,3,1,5,4,4,4,3,1,1,1,4,4,3,3,1,4,4,6,4,1,1,5,5,4,3,3,1,5,4,4,3,1,6,1,1,1]
    domain2=[4,4,4,5,1,4,4,4,4,4,4,4,4,3,3,4,1,4,4,4,1,3,1,3,1,4,4,4,4,1,1,3,1,1,4,3,4,4,5,4,2,3,3,1,1,5,4,4,5,4,4,1,1,4,4,5,4,3,3,4,1,4,4,3,3,1,5,4,4,4,5,4,4,5,1,4,4,4,4,5,5,3,3,4,4,1,4,3,1,5,4,4,4,3,1,1,1,4,4,3,3,1,4,4,1,4,1,1,5,5,4,3,3,1,5,4,4,3,1,1,1,1,1]

    lans=[]
    for i in language:
        lans.extend(i)
    kappa0=cohen_kappa_score(lans,lans)

    taskcodedoc1=[]
    taskcodedoc2=[]
    for i in tasktype1:
        if i in [1,2,3]:
            taskcodedoc1.append("code")
        else:
            taskcodedoc1.append("doc")
    for i in tasktype2:
        if i in [1,2,3]:
            taskcodedoc2.append("code")
        else:
            taskcodedoc2.append("doc")

    taskpac1=[]
    taskpac2=[]
    for i in domain1:
        if i in [1,4]:
            taskpac1.append("corrective")
        elif i in [2,5]:
            taskpac1.append("ada")
        else:
            taskpac1.append("perf")
    for i in domain2:
        if i in [1,4]:
            taskpac2.append("corrective")
        elif i in [2,5]:
            taskpac2.append("ada")
        else:
            taskpac2.append("perf")
    kappa1=cohen_kappa_score(taskcodedoc1,taskcodedoc2)
    kappa2=cohen_kappa_score(taskpac1,taskpac2)
    kappa3=cohen_kappa_score(domain1,domain2)
    print("kappa",kappa1,kappa2,kappa3)

    tasktypes=[]
    for i in tasktype:
        tasktypes.extend(i)
    print("tasktypecount",Counter(tasktypes))

    languages=[]
    for i in language:
        languages.extend(i)
    print("languagecount",Counter(languages))

    domains=[]
    for i in domain:
        domains.extend(i)
    print("domaincount",Counter(domains))


    figind=0
    fignames=['./figures/languageheatmap.png','./figures/tasktypeheatmap.png','./figures/domainheatmap.png']
    toptermslst=[['Python','JavaScript','TypeScript','C#','C++','PHP','Java'],[[1,4],[3,6],[2,5],[3,2,1],[6,5,4]],[4,1,3,5,2]]
    toptermsnamelst=[['Python','JavaScript','TypeScript','C#','C++','PHP','Java'],['Corrective','Perfective','Adaptive','Code','Doc'],['Non-web','Application','Web','Tools','System']]
    threedata=[language,tasktype,domain]
    ynames=['Ratio','Ratio','Ratio']
    labelname=["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31","32","33","34","35","36","37"]
    colors = ['b','r','g','darkorange','purple','c','y']
    for ind in [0,1,2]:
        topterms=toptermslst[ind]
        eachnewterm_gfi=threedata[ind]
        bt=[0]*len(eachnewterm_gfi)
        wlst=[]
        for i in range(len(topterms)):
            weightlst=[]
            for t in range(len(eachnewterm_gfi)):
                onenew=eachnewterm_gfi[t]
                if isinstance(topterms[i],list):
                    weight=0
                    for num in topterms[i]:
                        weight+=onenew.count(num)
                else:
                    weight=onenew.count(topterms[i])
                weightlst.append(weight/len(onenew))
            wlst.append(weightlst)
        wlst = sort_lsts(wlst)
        if ind==1:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4))
            for i in [0,1,2]:
                ax1.bar(list(range(len(labelname))),wlst[i],width=0.8,bottom=bt,label=toptermsnamelst[ind][i],color=colors[i])
                bt = [bt[m] + wlst[i][m] for m in range(len(bt))]
            ax1.legend(fontsize=12,loc="upper left")
            plt.xticks([])
            ax1.tick_params(labelsize=15)
            ax1.set_xlim((-1, 37))
            bwith = 2
            ax1.spines['bottom'].set_linewidth(bwith)
            ax1.spines['left'].set_linewidth(bwith)
            ax1.spines['top'].set_linewidth(bwith)
            ax1.spines['right'].set_linewidth(bwith)
            ax1.tick_params(bottom=False,top=False,left=False,right=False,labelsize=15, labelbottom=False)
            ax1.grid(alpha=0.6)
            plt.tight_layout()
            ax1.set_ylabel(ynames[ind],{'size':18})

            wlst = wlst[3:]
            wlst = sort_lsts(wlst)
            bt=[0]*len(eachnewterm_gfi)
            for i in [0,1]:
                ax2.bar(list(range(len(labelname))),wlst[i],width=0.8,bottom=bt,label=toptermsnamelst[ind][i+3],color=colors[i+3])
                bt = [bt[m] + wlst[i][m] for m in range(len(bt))]
            ax2.legend(fontsize=12,loc="upper left")
            plt.xlabel("Newcomers",{'size':18})
            plt.ylabel(ynames[ind],{'size':18})
            plt.xticks([])
            ax2.tick_params(labelsize=15)
            ax2.set_xlim((-1, 37))
            bwith = 2
            ax2.spines['bottom'].set_linewidth(bwith)
            ax2.spines['left'].set_linewidth(bwith)
            ax2.spines['top'].set_linewidth(bwith)
            ax2.spines['right'].set_linewidth(bwith)
            ax2.tick_params(bottom=False,top=False,left=False,right=False,labelsize=15)
            ax2.grid(alpha=0.6)
            plt.tight_layout()
            plt.savefig(fignames[ind], dpi=300)
        else:
            plt.figure(figsize=(8, 3))
            figind+=3
            for i in range(len(topterms)):
                plt.bar(list(range(len(labelname))),wlst[i],width=0.8,bottom=bt,label=toptermsnamelst[ind][i],color=colors[i])
                bt = [bt[m] + wlst[i][m] for m in range(len(bt))]
    
            plt.legend(fontsize=12,loc="upper left")
            plt.xlabel("Newcomers",{'size':18})
            plt.ylabel(ynames[ind],{'size':18})
            plt.xticks([])
            plt.xlim((-1, 37))
            ax = plt.gca()
            bwith = 2
            ax.spines['bottom'].set_linewidth(bwith)
            ax.spines['left'].set_linewidth(bwith)
            ax.spines['top'].set_linewidth(bwith)
            ax.spines['right'].set_linewidth(bwith)
            plt.tick_params(bottom=False,top=False,left=False,right=False,labelsize=15)
            plt.grid(alpha=0.6)
            plt.tight_layout()
            plt.savefig(fignames[ind], dpi=300)

    
    df = pd.read_pickle("./data/processeddata.pkl")
    
    dfnonew=df[df["cls_isnew"]==0]
    dfnew=df[df["cls_isnew"]==1]
    dfnew=dfnew.copy()
    dfnonew=dfnonew.copy()
    dfnew.reset_index(drop=True,inplace=True)
    dfnonew.reset_index(drop=True,inplace=True)


    allcls=dfnew["clsname"].values.tolist()
    clsproowner=dfnew["owner"].values.tolist()
    clsproname=dfnew["name"].values.tolist()
    clspros=[allcls[i]+' '+clsproowner[i]+' '+clsproname[i] for i in range(len(allcls))]
    clspros=list(set(clspros))
    clspros=[i.split(' ')[0] for i in clspros]


    a=Counter(clspros).most_common()
    b=a[0:100]
    c=len([i for i in a if i[1]>1])
    newcomers=[i[0] for i in b]
    counts=[i[1] for i in b]
    newcomers=newcomers[1:]
    counts=counts[1:]
    newcomers=[newcomers[i] for i in range(len(newcomers)) if counts[i]>2]

    newcomers=['misoguy','code-review-doctor','sameshl','faheel','accraze','rbrishabh','Bharat123rox','sainthkh','RhnSharma','timgates42','jamesgeorge007','raybellwaves','kentcdodds','phated',
        'minrk','cherniavskii','hs2361','sakulstra','willingc','markroth8','DickvdBrink','SanthoshBala18','flying-sheep','CharlesStover','aashil','arcanis','MarcoGorelli','Patil2099','naoyak',
        'marcosvega91','xjlim','imskr','FLGMwt','danedavid','Cheukting','saranshkataria','caugner']
    
    
    csv_newcomer=[]
    csv_url=[]
    csv_clst=[]

    
    csv_clsallcmt=[]
    csv_clsonemonth_cmt=[]
    csv_clstwomonth_cmt=[]
    csv_clsthreemonth_cmt=[]
    csv_clsissuesenmean=[]
    csv_clsissuesenmedian=[]
    csv_clsprsenmean=[]
    csv_clsprsenmedian=[]

    csv_clsallpr=[]
    csv_clsalliss=[]
    csv_clspronum=[]
    csv_clsiss=[]
    csv_clsallprreview=[]
    csv_clsonemonth_pr=[]
    csv_clstwomonth_pr=[]
    csv_clsthreemonth_pr=[]
    csv_clsonemonth_iss=[]

    csv_clstwomonth_iss=[]
    csv_clsthreemonth_iss=[]

    with open('./data/newcomerdata.json') as f:
        issuestr = json.load(f)
    issuedata = issuestr['0']
    lst=[]
    for i in range(len(issuedata)):
        lst.append(issuedata[str(i)])
    dfall=pd.DataFrame(lst)

    for m in newcomers:
        newdf=dfnew[dfnew["clsname"]==m]
        newowner=newdf["owner"].values.tolist()
        newname=newdf["name"].values.tolist()
        newclst=newdf["clst"].values.tolist()
        newclst=[datetime.fromtimestamp(x/1000) for x in newclst]

        newclstwomonth_cmt=newdf["clstwomonth_cmt"].values.tolist()
        newclsthreemonth_cmt=newdf["clsthreemonth_cmt"].values.tolist()
        newclsissuesenmedian=newdf["clsissuesenmedian"].values.tolist()
        newclsprsenmean=newdf["clsprsenmean"].values.tolist()
        newclsprsenmedian=newdf["clsprsenmedian"].values.tolist()

        newclsallpr=newdf["clsallpr"].values.tolist()
        newclsalliss=newdf["clsalliss"].values.tolist()
        newclspronum=newdf["clspronum"].values.tolist()
        newclsiss=newdf["clsiss"].values.tolist()
        newclsallprreview=newdf["clsallprreview"].values.tolist()
        newclsonemonth_pr=newdf["clsonemonth_pr"].values.tolist()
        newclstwomonth_pr=newdf["clstwomonth_pr"].values.tolist()
        newclsthreemonth_pr=newdf["clsthreemonth_pr"].values.tolist()
        newclsonemonth_iss=newdf["clsonemonth_iss"].values.tolist()

        newclstwomonth_iss=newdf["clstwomonth_iss"].values.tolist()
        newclsthreemonth_iss=newdf["clsthreemonth_iss"].values.tolist()


        pros=[]
        clstime=[]

        newclsallcmts=[]
        newclsonemonth_cmts=[]
        newclstwomonth_cmts=[]
        newclsthreemonth_cmts=[]
        newclsissuesenmeans=[]
        newclsissuesenmedians=[]
        newclsprsenmeans=[]
        newclsprsenmedians=[]

        newclsallprs=[]
        newclsallisss=[]
        newclspronums=[]
        newclsisss=[]
        newclsallprreviews=[]
        newclsonemonth_prs=[]
        newclstwomonth_prs=[]
        newclsthreemonth_prs=[]
        newclsonemonth_isss=[]

        newclstwomonth_isss=[]
        newclsthreemonth_isss=[]
        for i in range(len(newdf)):
            if newowner[i]+"/"+newname[i] in pros:
                a=pros.index(newowner[i]+"/"+newname[i])
                if clstime[a]>newclst[i]:
                    del clstime[a]
                    del pros[a]

                    del newclsallcmts[a]
                    del newclsonemonth_cmts[a]
                    del newclstwomonth_cmts[a]
                    del newclsthreemonth_cmts[a]
                    del newclsissuesenmeans[a]
                    del newclsissuesenmedians[a]
                    del newclsprsenmeans[a]
                    del newclsprsenmedians[a]

                    del newclsallprs[a]
                    del newclsallisss[a]
                    del newclspronums[a]
                    del newclsisss[a]
                    del newclsallprreviews[a]
                    del newclsonemonth_prs[a]
                    del newclstwomonth_prs[a]
                    del newclsthreemonth_prs[a]
                    del newclsonemonth_isss[a]

                    del newclstwomonth_isss[a]
                    del newclsthreemonth_isss[a]
                else:
                    continue

            pros.append(newowner[i]+"/"+newname[i])
            clstime.append(newclst[i])

            
            newclstwomonth_cmts.append(newclstwomonth_cmt[i])
            newclsthreemonth_cmts.append(newclsthreemonth_cmt[i])
            newclsissuesenmedians.append(newclsissuesenmedian[i])
            newclsprsenmeans.append(newclsprsenmean[i])
            newclsprsenmedians.append(newclsprsenmedian[i])

            newclsallprs.append(newclsallpr[i])
            newclsallisss.append(newclsalliss[i])
            newclspronums.append(newclspronum[i])
            newclsisss.append(newclsiss[i])
            newclsallprreviews.append(newclsallprreview[i])
            newclsonemonth_prs.append(newclsonemonth_pr[i])
            newclstwomonth_prs.append(newclstwomonth_pr[i])
            newclsthreemonth_prs.append(newclsthreemonth_pr[i])
            newclsonemonth_isss.append(newclsonemonth_iss[i])

            newclstwomonth_isss.append(newclstwomonth_iss[i])
            newclsthreemonth_isss.append(newclsthreemonth_iss[i])


            issuetitles,issuebodys,userallcmt,onemonth_cmt=get_userdata(m,i)
            newclsallcmts.append(userallcmt)
            newclsonemonth_cmts.append(onemonth_cmt)
            newclsissuesenmeans.append(cls_issue_sen_mean(issuetitles,issuebodys))

        csv_newcomer.extend([m]*(len(pros)))

        csv_clsallcmt.extend(newclsallcmts)
        csv_clsonemonth_cmt.extend(newclsonemonth_cmts)
        csv_clstwomonth_cmt.extend(newclstwomonth_cmts)
        csv_clsthreemonth_cmt.extend(newclsthreemonth_cmts)
        csv_clsissuesenmean.extend(newclsissuesenmeans)
        csv_clsissuesenmedian.extend(newclsissuesenmedians)
        csv_clsprsenmean.extend(newclsprsenmeans)
        csv_clsprsenmedian.extend(newclsprsenmedians)

        csv_clsallpr.extend(newclsallprs)
        csv_clsalliss.extend(newclsallisss)
        csv_clspronum.extend(newclspronums)
        csv_clsiss.extend(newclsisss)
        csv_clsallprreview.extend(newclsallprreviews)
        csv_clsonemonth_pr.extend(newclsonemonth_prs)
        csv_clstwomonth_pr.extend(newclstwomonth_prs)
        csv_clsthreemonth_pr.extend(newclsthreemonth_prs)
        csv_clsonemonth_iss.extend(newclsonemonth_isss)

        csv_clstwomonth_iss.extend(newclstwomonth_isss)
        csv_clsthreemonth_iss.extend(newclsthreemonth_isss)

    a=Counter(csv_newcomer).most_common()
    newcomers=[i[0] for i in a]
    csv_newcomer_=[]
    csv_dif_=[]

    csv_clsallcmt_=[]
    csv_clsonemonth_cmt_=[]
    csv_clstwomonth_cmt_=[]
    csv_clsthreemonth_cmt_=[]
    csv_clsissuesenmean_=[]
    csv_clsissuesenmedian_=[]
    csv_clsprsenmean_=[]
    csv_clsprsenmedian_=[]

    csv_clsallpr_=[]
    csv_clsalliss_=[]
    csv_clspronum_=[]
    csv_clsiss_=[]
    csv_clsallprreview_=[]
    csv_clsonemonth_pr_=[]
    csv_clstwomonth_pr_=[]
    csv_clsthreemonth_pr_=[]
    csv_clsonemonth_iss_=[]

    csv_clstwomonth_iss_=[]
    csv_clsthreemonth_iss_=[]

    for i in newcomers:
        count=csv_newcomer.count(i)
        index_list = []
        index = -1
        for m in range(0, count):
            index = csv_newcomer.index(i, index + 1)
            index_list.append(index)
        for t in index_list:
            csv_newcomer_.append(csv_newcomer[t])

            csv_clsallcmt_.append(csv_clsallcmt[t])
            csv_clsonemonth_cmt_.append(csv_clsonemonth_cmt[t])
            csv_clstwomonth_cmt_.append(csv_clstwomonth_cmt[t])
            csv_clsthreemonth_cmt_.append(csv_clsthreemonth_cmt[t])
            csv_clsissuesenmean_.append(csv_clsissuesenmean[t])
            csv_clsissuesenmedian_.append(csv_clsissuesenmedian[t])
            csv_clsprsenmean_.append(csv_clsprsenmean[t])
            csv_clsprsenmedian_.append(csv_clsprsenmedian[t])

            csv_clsallpr_.append(csv_clsallpr[t])
            csv_clsalliss_.append(csv_clsalliss[t])
            csv_clspronum_.append(csv_clspronum[t])
            csv_clsiss_.append(csv_clsiss[t])
            csv_clsallprreview_.append(csv_clsallprreview[t])
            csv_clsonemonth_pr_.append(csv_clsonemonth_pr[t])
            csv_clstwomonth_pr_.append(csv_clstwomonth_pr[t])
            csv_clsthreemonth_pr_.append(csv_clsthreemonth_pr[t])
            csv_clsonemonth_iss_.append(csv_clsonemonth_iss[t])

            csv_clstwomonth_iss_.append(csv_clstwomonth_iss[t])
            csv_clsthreemonth_iss_.append(csv_clsthreemonth_iss[t])


    npfis=[]
    nonnpfis=[]
    for n in newcomers:
        dfnewi=dfnew[dfnew["clsname"]==n]
        dfnonewi=dfnonew[dfnonew["clsname"]==n]

        newclst=dfnewi["clst"].values.tolist()
        newowner=dfnewi["owner"].values.tolist()
        newname=dfnewi["name"].values.tolist()
        newissemb=dfnewi["issueemb"].values.tolist()
        newisstitle=dfnewi["title"].values.tolist()
        newissbody=dfnewi["body"].values.tolist()
        newprodesemb=dfnewi["prodesemb"].values.tolist()
        newprodescription=dfnewi["prodescription"].values.tolist()
        newisstext=[newisstitle[x]+newissbody[x] for x in range(len(newisstitle))]
        
        nonewissemb=dfnonewi["issueemb"].values.tolist()
        nonewisstitle=dfnonewi["title"].values.tolist()
        nonewissbody=dfnonewi["body"].values.tolist()
        nonewisst=dfnonewi["clst"].values.tolist()
        nonewprodesemb=dfnonewi["prodesemb"].values.tolist()
        nonewprodescription=dfnonewi["prodescription"].values.tolist()
        nonewisstext=[nonewisstitle[x]+nonewissbody[x] for x in range(len(nonewisstitle))]

        pros=[]
        newt=[]
        newemb=[]
        nonewemb=[]
        newtext=[]
        nonewtext=[]

        newdesemb=[]
        nonewdesemb=[]
        newdescription=[]
        nonewdescription=[]

        nonewt=[]
        for i in range(len(newissemb)):
            if newowner[i]+"/"+newname[i] in pros:
                a=pros.index(newowner[i]+"/"+newname[i])
                if newt[a]>newclst[i]:
                    nonewemb.append(newemb[a])
                    nonewtext.append(newtext[a])
                    nonewdesemb.append(newdesemb[a])
                    nonewdescription.append(newdescription[a])
                    nonewt.append(newt[a])
                    del newt[a]
                    del pros[a]
                    del newemb[a]
                    del newtext[a]
                    del newdesemb[a]
                    del newdescription[a]
                else:
                    continue
            newemb.append(newissemb[i])
            newtext.append(newisstext[i])
            newdesemb.append(newprodesemb[i])
            newdescription.append(newprodescription[i])
            pros.append(newowner[i]+"/"+newname[i])
            newt.append(newclst[i])
        nonewemb=nonewemb+nonewissemb
        nonewtext=nonewtext+nonewisstext
        nonewdesemb=nonewdesemb+nonewprodesemb
        nonewdescription=nonewdescription+nonewprodescription
        nonewt=nonewt+nonewisst

        addnew=[[newemb[i],newtext[i],newt[i],newdesemb[i],newdescription[i]] for i in range(len(newemb))]
        addnonew=[[nonewemb[i],nonewtext[i],nonewt[i],nonewdesemb[i],nonewdescription[i]] for i in range(len(nonewemb))]
        npfis.append(addnew)
        nonnpfis.append(addnonew)

    selfhis_cos=[]
    selfhis_jac=[]
    selfhis_dom_cos=[]
    selfhis_dom_jac=[]
    for iss in npfis:
        for i in range(len(iss)):
            aiss=iss[i]
            for m in range(len(iss)-i-1):
                biss=iss[i+1+m]
                if aiss[2]>biss[2]:
                    selfhis_cos.append(cos_sim(aiss[0],biss[0]))
                    selfhis_jac.append(jaccard_sim(aiss[1],biss[1]))
                    selfhis_dom_cos.append(cos_sim(aiss[3],biss[3]))
                    selfhis_dom_jac.append(jaccard_sim(aiss[4],biss[4]))
    for i in range(len(npfis)):
        for aiss in npfis[i]:
            for biss in nonnpfis[i]:
                if aiss[2]>biss[2]:
                    selfhis_cos.append(cos_sim(aiss[0],biss[0]))
                    selfhis_jac.append(jaccard_sim(aiss[1],biss[1]))
                    selfhis_dom_cos.append(cos_sim(aiss[3],biss[3]))
                    selfhis_dom_jac.append(jaccard_sim(aiss[4],biss[4]))

    selfhis=[selfhis_jac,selfhis_cos,selfhis_dom_jac,selfhis_dom_cos]

    otherNPs_cos=[]
    otherNPs_jac=[]
    otherNPs_dom_cos=[]
    otherNPs_dom_jac=[]
    for i in range(len(npfis)):
        for m in range(len(npfis)-i-1):
            for aiss in npfis[i]:
                for biss in npfis[i+1+m]:
                    otherNPs_cos.append(cos_sim(aiss[0],biss[0]))
                    otherNPs_jac.append(jaccard_sim(aiss[1],biss[1]))
                    otherNPs_dom_cos.append(cos_sim(aiss[3],biss[3]))
                    otherNPs_dom_jac.append(jaccard_sim(aiss[4],biss[4]))
    otherNPs=[otherNPs_jac,otherNPs_cos,otherNPs_dom_jac,otherNPs_dom_cos]

    pos=0
    plt.figure(figind,figsize=(8, 4))
    figind+=1
    for i in range(4): 
        bplot=plt.boxplot([selfhis[i],otherNPs[i]],widths=0.8,
        positions=[pos,pos+1], showmeans=True,showfliers=False,patch_artist=True,meanprops = {'markerfacecolor':'black','markeredgecolor':'black', 'markersize':10})
        [[item.set_color('k') for item in bplot['medians']]]
        colors = ['b','r']
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        pos+=3

    plt.tick_params(bottom=False,top=False,left=False,right=False,labelsize=15)
    plt.xlim((-1, 11))
    plt.ylabel('Similarity',{'size':18})
    plt.xticks([0.5,3.5,6.5,9.5],["issue_cos","issue_jac","desc_cos","desc_jac"])
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(handles=[bplot["boxes"][0],bplot["boxes"][1]],labels=["FIs compared with own historical issues","FIs compared with others' FIs"],fontsize=12,loc="upper left")
    ax = plt.gca()
    bwith = 2
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    plt.grid(alpha=0.6)
    plt.tight_layout()
    plt.savefig('./figures/similarNP.png')               


    csv_type_=[3,3,3,1,3,1,1,1,1,4,6,5,6,4,4,1,1,6,6,6,6,1,4,1,1,4,4,4,6,1,2,1,3,2,4,4,4,4,4,4,4,6,6,3,4,6,6,6,4,4,1,3,1,3,1,1,1,1,1,1,1,4,4,6,1,1,6,6,6,1,1,1,1,1,1,4,1,4,6,3,3,1,3,4,6,1,6,1,1,1,6,1,1,3,3,1,2,6,1,1,6,6,1,1,4,3,6,1,6,1,1,1,4,3,6,6,6,1,1,1,6,3,3]
    csv_domain_=[4,4,4,5,1,4,4,4,4,4,4,4,4,3,3,4,1,4,4,4,1,3,1,3,1,4,4,4,4,1,1,3,1,1,4,3,4,4,5,4,2,3,3,1,1,1,4,4,5,4,4,1,1,4,4,5,4,3,3,4,1,4,4,3,3,1,5,4,4,4,1,4,4,5,1,4,4,4,4,5,1,3,3,4,4,1,4,3,1,5,4,4,4,3,1,1,1,4,4,3,3,1,4,4,1,4,1,1,5,5,4,3,3,1,5,4,4,3,1,1,1,1,1]
    csv_clsonemonth_cmt_=[i*10 for i in csv_clsonemonth_cmt_]
    csv_clsissuesenmean_=[i*1000 for i in csv_clsissuesenmean_]

    csv_clsallprreview_=[i*10 for i in csv_clsallprreview_]
    csv_clsonemonth_pr_=[i*10 for i in csv_clsonemonth_pr_]
    csv_clsallpr_=[i*10 for i in csv_clsallpr_]
    csv_clsalliss_=[i*10 for i in csv_clsalliss_]
    csv_clsissuesenmedian_=[i*100 for i in csv_clsissuesenmedian_]
    csv_clsprsenmean_=[i*100 for i in csv_clsprsenmean_]
    csv_clsprsenmedian_=[i*100 for i in csv_clsprsenmedian_]

    contentslst=[[csv_clsallcmt_,csv_clsonemonth_cmt_,csv_clsissuesenmean_],[csv_clsallpr_,csv_clsalliss_,csv_clspronum_],[csv_clsallprreview_, csv_clstwomonth_cmt_, csv_clsthreemonth_cmt_, csv_clsonemonth_pr_],[csv_clstwomonth_pr_, csv_clsthreemonth_pr_, csv_clsonemonth_iss_, csv_clstwomonth_iss_],[csv_clsthreemonth_iss_, csv_clsissuesenmedian_,csv_clsprsenmean_, csv_clsprsenmedian_]]
    
    for x in range(5):
        csv_contents=contentslst[x]
        plt.figure(figind,figsize=(8.4, 4))
        ax = plt.axes(polar=False)
        figind+=1
        pos=0
        for content in csv_contents:
            list1=[]
            list2=[]
            list3=[]
            list4=[]
            list5=[]
            list6=[]
            for i in range(len(csv_type_)):
                if csv_type_[i]==1:
                    list1.append(content[i])
                    list3.append(content[i])
                elif csv_type_[i]==2:
                    list1.append(content[i])
                    list4.append(content[i])
                elif csv_type_[i]==3:
                    list1.append(content[i])
                    list5.append(content[i])
                elif csv_type_[i]==4:
                    list2.append(content[i])
                    list3.append(content[i])
                elif csv_type_[i]==5:
                    list2.append(content[i])
                    list4.append(content[i])
                elif csv_type_[i]==6:
                    list2.append(content[i])
                    list5.append(content[i])
            bplot=plt.boxplot([list1,list2,list3,list4,list5],widths=0.8,
            positions=[pos,pos+1,pos+2,pos+3,pos+4], showmeans=True,showfliers=False,patch_artist=True,meanprops = {'markerfacecolor':'black','markeredgecolor':'black', 'markersize':10})
            [[item.set_color('k') for item in bplot['medians']]]
            colors = ['b','r','g','darkorange','purple']
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
            pos+=6
        plt.tick_params(bottom=False,top=False,left=False,right=False,labelsize=15)
        if x==0:
            plt.xticks([2,8,14],['#cmt_all',r"#cmt_onem$\times$10",r"iss_polar_ave$\times$10$^{3}$"])
        elif x==1:
            plt.xticks([2,8,14],[r'#pr_all$\times$10',r"#iss_all$\times$10","#pro"])
        elif x==2:
            plt.xticks([2,8,14,20],[r'#prreview_all$\times$10',"#cmt_twom","#cmt_threem",r"#pr_onem$\times$10"])
        elif x==3:
            plt.xticks([2,8,14,20],["#pr_twom","#pr_threem","#iss_onem","#iss_twom"])
        elif x==4:
            plt.xticks([2,8,14,20],["#iss_threem",r"iss_polar_med$\times$10$^{2}$",r"pr_polar_ave$\times$10$^{2}$",r"pr_polar_med$\times$10$^{2}$"])
        plt.ylabel('Values',{'size':18})
        if x==0 or x==1:
            plt.xlim((-1, 17))
            plt.xticks(fontsize=18)
        elif x==2 or x==3:
            plt.xlim((-1, 23))
            plt.xticks(fontsize=18)
        elif x==4:
            plt.xlim((-1, 23))
            plt.xticks(fontsize=12)
        plt.yticks(fontsize=18)
        plt.legend(handles=[bplot["boxes"][0],bplot["boxes"][1],bplot["boxes"][2],bplot["boxes"][3],bplot["boxes"][4]],labels=["Code","Doc","Corrective","Adaptive","Perfectiv"],fontsize=12,loc="upper right")
        ax = plt.gca()
        bwith = 2
        ax.spines['bottom'].set_linewidth(bwith)
        ax.spines['left'].set_linewidth(bwith)
        ax.spines['top'].set_linewidth(bwith)
        ax.spines['right'].set_linewidth(bwith)
        plt.grid(alpha=0.6)
        plt.tight_layout()
        plt.savefig('./figures/typedis_'+str(x)+'.png')


        pos=0
        plt.figure(figind,figsize=(8, 4))
        ax = plt.axes(polar=False)
        figind+=1
        for content in csv_contents:
            list1=[]
            list2=[]
            list3=[]
            list4=[]
            list5=[]
            for i in range(len(csv_domain_)):
                if csv_domain_[i]==1:
                    list1.append(content[i])
                elif csv_domain_[i]==2:
                    list2.append(content[i])
                elif csv_domain_[i]==3:
                    list3.append(content[i])
                elif csv_domain_[i]==4:
                    list4.append(content[i])
                elif csv_domain_[i]==5:
                    list5.append(content[i])
            bplot=plt.boxplot([list1,list2,list3,list4,list5],widths=0.8,
            positions=[pos,pos+1,pos+2,pos+3,pos+4], showmeans=True,showfliers=False,patch_artist=True,meanprops = {'markerfacecolor':'black','markeredgecolor':'black', 'markersize':10})
            [[item.set_color('k') for item in bplot['medians']]]
            colors = ['b','r','g','darkorange','purple']
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
            pos+=6
        plt.tick_params(bottom=False,top=False,left=False,right=False,labelsize=15)
        if x==0:
            plt.xticks([2,8,14],['#cmt_all',r"#cmt_onem$\times$10",r"iss_polar_ave$\times$10$^{3}$"])
        elif x==1:
            plt.xticks([2,8,14],[r'#pr_all$\times$10',r"#iss_all$\times$10","#pro"])
        elif x==2:
            plt.xticks([2,8,14,20],[r'#prreview_all$\times$10',"#cmt_twom","#cmt_threem",r"#pr_onem$\times$10"])
        elif x==3:
            plt.xticks([2,8,14,20],["#pr_twom","#pr_threem","#iss_onem","#iss_twom"])
        elif x==4:
            plt.xticks([2,8,14,20],["#iss_threem",r"iss_polar_med$\times$10$^{2}$",r"pr_polar_ave$\times$10$^{2}$",r"pr_polar_med$\times$10$^{2}$"])
        plt.ylabel('Values',{'size':18})
        if x==0 or 1:
            plt.xlim((-1, 17))
            plt.xticks(fontsize=18)
        else:
            plt.xlim((-1, 23))
            plt.xticks(fontsize=14)

        plt.yticks(fontsize=18)
        plt.legend(handles=[bplot["boxes"][0],bplot["boxes"][1],bplot["boxes"][2],bplot["boxes"][3],bplot["boxes"][4]],labels=["Application", "System", "Web", "Non-web", "Tools"],fontsize=12,loc="upper right")
        ax = plt.gca()
        bwith = 2
        ax.spines['bottom'].set_linewidth(bwith)
        ax.spines['left'].set_linewidth(bwith)
        ax.spines['top'].set_linewidth(bwith)
        ax.spines['right'].set_linewidth(bwith)
        plt.grid(alpha=0.6)
        plt.tight_layout()
        plt.savefig('./figures/domaindis_'+str(x)+'.png')
    print("finish")