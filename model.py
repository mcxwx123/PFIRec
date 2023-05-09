import pandas as pd
import lightgbm
import numpy as np
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import random
import math
import json
import matplotlib.pyplot as plt

def get_proname(owner,name):
    return owner+"/"+name

def combine(title,body):
    return title+body

def norm_score(scores,max_score,min_score):
    if max_score == min_score:
        norm_sore_list=[1.0 for score in scores]
    else:
        norm_sore_list=[1.0 * (score - min_score) / (max_score - min_score) for score in scores]
    return np.asarray([norm_sore_list])

def gettitle(owner, name, number):
    selected_rows = dfall.loc[(dfall['owner'] == owner) & (dfall['name']==name)& (dfall['number']==number)]
    return selected_rows.iloc[0]['title']

def getbody(owner, name, number):
    selected_rows = dfall.loc[(dfall['owner'] == owner) & (dfall['name']==name)& (dfall['number']==number)]
    return selected_rows.iloc[0]['body']


def get_lamb_metrics(training_set,valid_set,test_set,idname,xnames):
    qids_train = training_set.groupby(idname)[idname].count().to_numpy()
    X_train = training_set[xnames]
    y_train = training_set[["match"]]
    
    qids_validation = valid_set.groupby(idname)[idname].count().to_numpy()
    X_validation = valid_set[xnames]
    y_validation = valid_set[["match"]]

    model = lightgbm.LGBMRanker(
    objective="lambdarank",
    )

    model.fit(
    X=X_train,
    y=y_train,
    group=qids_train,
    eval_set=[(X_validation, y_validation)],
    eval_group=[qids_validation],
    eval_at=5,
    )
    metrics=get_metrics(model,idname,test_set,xnames,0)
    return metrics

def get_XGB_metrics(training_set,valid_set,test_set,idname,xnames):
    X_train = training_set[xnames]
    y_train = training_set[["match"]]
    
    X_validation = valid_set[xnames]
    y_validation = valid_set[["match"]]

    X_train=pd.concat([X_train,X_validation],ignore_index=True)
    y_train=pd.concat([y_train,y_validation],ignore_index=True)

    model = XGBClassifier(eval_metric=['logloss','auc','error'],use_label_encoder=False) 
    model.fit(X_train,y_train)

    metrics=get_metrics(model,idname,test_set,xnames,1)
    return metrics

def get_Logistic_regression_metrics(training_set,valid_set,test_set,idname,xnames):
    X_train = training_set[xnames]
    y_train = training_set[["match"]]
    
    X_validation = valid_set[xnames]
    y_validation = valid_set[["match"]]

    X_train=pd.concat([X_train,X_validation],ignore_index=True)
    y_train=pd.concat([y_train,y_validation],ignore_index=True)

    model = LogisticRegression(max_iter=5000,random_state=0)
    y_train=y_train["match"].values.tolist()
    model.fit(X_train,y_train)

    metrics=get_metrics(model,idname,test_set,xnames,1)
    return metrics

def get_Random_forest_metrics(training_set,valid_set,test_set,idname,xnames):
    X_train = training_set[xnames]
    y_train = training_set[["match"]]
    
    X_validation = valid_set[xnames]
    y_validation = valid_set[["match"]]

    X_train=pd.concat([X_train,X_validation],ignore_index=True)
    y_train=pd.concat([y_train,y_validation],ignore_index=True)

    model = RandomForestClassifier(n_estimators=10, criterion='gini',random_state=0)
    y_train=y_train["match"].values.tolist()
    model.fit(X_train,y_train)

    metrics=get_metrics(model,idname,test_set,xnames,1)
    return metrics

def get_recgfi_metrics(training_set_recgfi,test_set,idname,xnames):
    p_train = training_set_recgfi[training_set_recgfi.cls_isnew == 1]
    n_train = training_set_recgfi[training_set_recgfi.cls_isnew == 0]
    p_train = p_train.sample(frac=n_train.shape[0]/p_train.shape[0],replace=True,random_state=0)
    training_set_recgfi=pd.concat([p_train,n_train],ignore_index=True)
    training_set_recgfi=training_set_recgfi.sample(frac=1, random_state=0)

    X_train = training_set_recgfi[xnames]
    y_train = training_set_recgfi["cls_isnew"]
    model = XGBClassifier(eval_metric=['logloss','auc','error'],use_label_encoder=False)
    model.fit(X_train,y_train)
    metrics=get_metrics(model,idname,test_set,xnames,1)
    return metrics


def get_Stanik_metrics(training_set_recgfi,test_set,idname,xnames):
    p_train = training_set_recgfi[training_set_recgfi.cls_isnew == 1]
    n_train = training_set_recgfi[training_set_recgfi.cls_isnew == 0]
    p_train = p_train.sample(frac=n_train.shape[0]/p_train.shape[0],replace=True,random_state=0)
    training_set_recgfi=pd.concat([p_train,n_train],ignore_index=True)
    training_set_recgfi=training_set_recgfi.sample(frac=1, random_state=0)

    X_train = training_set_recgfi[xnames]
    vectorizer = TfidfVectorizer(analyzer='word', input='content', stop_words='english', max_features=50)
    X_train=X_train.copy()
    X_train['texts'] = X_train.apply(lambda row: combine(row['title'], row['body']), axis=1)
    del X_train['title']
    del X_train['body']
    texts = pd.DataFrame(vectorizer.fit_transform(X_train['texts'].values).toarray())
    issue_con=X_train[["LengthOfTitle","LengthOfDescription","issuesen"]]
    T=pd.concat([texts,issue_con],axis=1)
    c=[]
    for i in range(texts.shape[1]):
        c.append("texts"+str(i))
    c+=list(issue_con.columns)
    T.columns=c
    X_train=T
    
    y_train = training_set_recgfi["cls_isnew"]
    model = RandomForestClassifier(n_estimators=10, criterion='gini',random_state=0)
    model.fit(T,y_train)
    metrics=get_metrics(model,idname,test_set,xnames,2)
    return metrics

def get_random_metrics(test_set,idname):
    metrics=get_metrics(None,idname,test_set,['LengthOfTitle'],3)
    return metrics

def get_gfilabel_metrics(test_set,idname):
    metrics=get_metrics(None,idname,test_set,['issuet','gfilabelnum'],4)
    return metrics

def get_timesort_metrics(test_set,idname):
    metrics=get_metrics(None,idname,test_set,['issuet'],5)
    return metrics


def toisgfi(gfilabelnum):
    return int(gfilabelnum>0)

def get_metrics(model,idname,test_set,xnames,isxgb):
    global randseed
    top1match=[]
    top3match=[]
    top5match=[]
    top10match=[]
    firstrank=[]
    list_ids=list(set(test_set[idname].values.tolist()))
    list_ids.sort()
    
    true_relevance_list=[]
    scores_list=[]
    max_score=0
    min_score=0
    for id in list_ids:
        clsset=test_set[test_set[idname]==id]
        clsset.reset_index(drop=True,inplace=True)
        clsset=clsset.copy()
        if 'title' in xnames:
            clsset['title'] = clsset.apply(lambda row: gettitle(row['owner'], row['name'], row['number']), axis=1)
            clsset['body'] = clsset.apply(lambda row: getbody(row['owner'], row['name'], row['number']), axis=1)
        X_test=clsset[xnames] 
        if isxgb==2:
            vectorizer = TfidfVectorizer(analyzer='word', input='content', stop_words='english', max_features=50)
            X_test=X_test.copy()
            X_test['texts'] = X_test.apply(lambda row: combine(row['title'], row['body']), axis=1)
            del X_test['title']
            del X_test['body']
            texts = pd.DataFrame(vectorizer.fit_transform(X_test['texts'].values).toarray())
            issue_con=X_test[["LengthOfTitle","LengthOfDescription","issuesen"]]
            T=pd.concat([texts,issue_con],axis=1)
            c=[]
            for i in range(texts.shape[1]):
                c.append("texts"+str(i))
            c+=list(issue_con.columns)
            T.columns=c
            X_test=T
        X_test.reset_index(drop=True,inplace=True)

        true_relevance=np.asarray([clsset["match"].values.tolist()])
        if isxgb==0:
            scores=list(model.predict(X_test))
            max_score=max(scores+[max_score])
            min_score=min(scores+[min_score])
        elif isxgb==1 or isxgb==2:
            scores=np.asarray([list(model.predict_proba(X_test)[:,1])])
        elif isxgb==3:
            random.seed(randseed)
            randseed+=1
            randlst=random.sample(range(len(X_test)), len(X_test))
            randlst=[i/(len(X_test)-1) for i in randlst]
            scores=np.asarray([randlst])
        elif isxgb==4:
            X_test=X_test.copy()
            X_test_gfilabnum=X_test['gfilabelnum'].values.tolist()
            X_test.reset_index(drop=True,inplace=True)
            scores=[]
            random.seed(randseed)
            randseed+=1
            randlst=random.sample(range(len(X_test)), len(X_test))
            randlst=[i/(len(X_test)-1) for i in randlst]
            for k in range(len(X_test)):
                if X_test_gfilabnum[k]>0:
                    scores.append(randlst[k]+2)
                else:
                    scores.append(randlst[k])
            max_score=max(scores+[max_score])
            min_score=min(scores+[min_score])
        elif isxgb==5:
            X_testsort=X_test.sort_values(by=['issuet'],ascending=[False])
            indlst=X_testsort.index.tolist()
            scores=[]
            for k in range(len(X_test)):
                scores.append(1-indlst.index(k)/(len(X_test)-1))
            scores=np.asarray([scores])
        true_relevance_list.append(true_relevance)
        scores_list.append(scores)

    if isxgb in [0,4]:
        new_scores_list=[norm_score(s,max_score,min_score) for s in scores_list]
    else:
        new_scores_list=scores_list
    for i in range(len(new_scores_list)):
        true_relevance=true_relevance_list[i]
        if len(true_relevance[0])<10:
            continue
        scores=new_scores_list[i]

        matchrank=rank(true_relevance,scores)
        top1match.append(int(sum([i<1 for i in matchrank])>0))
        top3match.append(int(sum([i<3 for i in matchrank])>0))
        top5match.append(int(sum([i<5 for i in matchrank])>0))
        top10match.append(int(sum([i<10 for i in matchrank])>0))
        firstrank.append(min(matchrank))

    ratio1=np.mean(top1match)
    ratio3=np.mean(top3match)
    ratio5=np.mean(top5match)
    ratio10=np.mean(top10match)

    firstrank=[i+1 for i in firstrank]
    firsthitmedian=np.median(firstrank)
    return [ratio1,ratio3,ratio5,ratio10,firsthitmedian]



def get_similarityrank(idname,dataset):
    firstrank=[]
    allrank=[]
    list_ids=list(set(dataset[idname].values.tolist()))
    list_ids.sort()
    for simname in ['issjaccard_sim','cmtjaccard_sim',"prjaccard_sim",'issjaccard_sim_mean','cmtjaccard_sim_mean','prjaccard_sim_mean']:
        for id in list_ids:
            clsset=dataset[dataset[idname]==id]
            clsset.reset_index(drop=True,inplace=True)
            true_relevance=np.asarray([clsset["match"].values.tolist()])
            scores=np.asarray([clsset[simname].values.tolist()])
            matchrank=rank(true_relevance,scores)
            allrank.extend(matchrank)
            firstrank.append(min(matchrank))

        allrank=[i+1 for i in allrank]
        allmean=np.mean(allrank)
        allmedian=np.median(allrank)
        firstrank=[i+1 for i in firstrank]
        firstmean=np.mean(firstrank)
        firstmedian=np.median(firstrank)
        print(simname, allmean,allmedian,firstmedian,firstmean)

def rank(a,b):
    a=list(a[0])
    b=list(b[0])
    indlst=[]
    while 1 in a:
        ind=a.index(1)
        indlst.append(ind)
        a[ind]=0
    sortlst=[b[i] for i in indlst]
    b.sort()
    res0=[len(b)-b.index(i)-1 for i in sortlst]
    b.sort(reverse = True)
    res1=[b.index(i) for i in sortlst]
    res=[math.floor((res0[i]+res1[i])/2) for i in range(len(res0))]
    return res

def ifdelete(procmt):
    if procmt==procmt:
        return 0
    else:
        return 1

if __name__ == "__main__":
    idname="issgroupid"
    xnames_sub_cumu=[
                    #General OSS experience
                    "clsallcmt","clsallpr","clsalliss","clspronum","clsiss",'clsallprreview',]
    
    xnames_sub_act=[
                    #Activeness
                    'clsonemonth_cmt', 'clstwomonth_cmt', 'clsthreemonth_cmt', 'clsonemonth_pr', 'clstwomonth_pr', 'clsthreemonth_pr', 'clsonemonth_iss', 'clstwomonth_iss', 'clsthreemonth_iss',]

    xnames_sub_sen=[
                    #Sentiment
                    'clsissuesenmean', 'clsissuesenmedian','clsprsenmean', 'clsprsenmedian',]
    
    xnames_sub_clssolvediss=[##Expertise preference
                    #Content preference
                    'solvedisscos_sim','solvedisscos_mean',
                    "solvedissjaccard_sim","solvedissjaccard_sim_mean",
                    "solvedissuelabel_sum","solvedissuelabel_ratio",
                    ]
    
    xnames_sub_clsrptiss=[##Expertise preference
                    #Content preference
                    'issjaccard_sim','issjaccard_sim_mean',
                    "isscos_sim","isscos_mean",
                    'issuelabel_sum','issuelabel_ratio',
                    ]
    
    xnames_sub_clscomtiss=[##Expertise preference
                    #Content preference
                    'commentissuelabel_sum','commentissuelabel_ratio',
                    'commentissuecos_sim','commentissuecos_sim_mean',
                    'commentissuejaccard_sim','commentissuejaccard_sim_mean',
                    ]
    
    xnames_sub_clscmt=[##Expertise preference
                    #Content preference
                    'cmtjaccard_sim','cmtjaccard_sim_mean',
                    "cmtcos_sim","cmtcos_mean",
                    #"cmteuc_sim", "cmteuc_mean",
                    ]
    
    xnames_sub_clspr=[##Expertise preference
                    #Content preference
                    "prjaccard_sim",'prjaccard_sim_mean',
                    "prcos_sim","prcos_mean",
                    'prlabel_sum','prlabel_ratio',
                    ]
    
    xnames_sub_clsprreview=[##Expertise preference
                    #Content preference
                    'prreviewcos_sim', 'prreviewcos_sim_mean', 
                    'prreviewjaccard_sim', 'prreviewjaccard_sim_mean',
                    ]

    xnames_sub_clscont=xnames_sub_clscmt+xnames_sub_clspr+xnames_sub_clsprreview+xnames_sub_clsrptiss+xnames_sub_clscomtiss+xnames_sub_clssolvediss+['lan_sim']
    
    xnames_sub_domain=[##Expertise preference
                        #Domain preference
                        'readmecos_sim_mean', 'readmecos_sim',
                        'readmejaccard_sim_mean', 'readmejaccard_sim',

                        "procos_mean","procos_sim",
                        "projaccard_mean",'projaccard_sim',
                        'prostopic_sum', 'prostopic_ratio',
                        ]
    
    xnames_sub_isscont=[##Candidate issues
                       #Content of issues
                      'LengthOfTitle', 'LengthOfDescription', 'NumOfCode', 'NumOfUrls', 'NumOfPics', 'buglabelnum', 'featurelabelnum', 'testlabelnum', 'buildlabelnum', 'doclabelnum', 'codinglabelnum', 'enhancelabelnum','gfilabelnum', 'mediumlabelnum', 'majorlabelnum', 'triagedlabelnum', 'untriagedlabelnum', 'labelnum',
                      'issuesen', 'coleman_liau_index', 'flesch_reading_ease','flesch_kincaid_grade','automated_readability_index',
                      ]

    xnames_sub_back=[##Candidate issues
                        #Background of issues
                        'pro_gfi_ratio','pro_gfi_num', 'proclspr', 'crtclsissnum', 'pro_star','openiss', 'openissratio','contributornum','procmt',
                        'rptcmt', 'rptiss', 'rptpr', 'rptpronum', 'rptallcmt', 'rptalliss', 'rptallpr', 'rpt_reviews_num_all', 'rpt_max_stars_commit', 'rpt_max_stars_issue', 'rpt_max_stars_pull', 'rpt_max_stars_review', 'rptisnew', 'rpt_gfi_ratio', #
                        'ownercmt','owneriss', 'ownerpr', 'ownerpronum', 'ownerallcmt', 'owneralliss', 'ownerallpr', 'owner_reviews_num_all', 'owner_max_stars_commit', 'owner_max_stars_issue', 'owner_max_stars_pull', 'owner_max_stars_review', 'owner_gfi_ratio', 'owner_gfi_num',
                        ]


    xnames_LambdaMART=xnames_sub_cumu+xnames_sub_act+xnames_sub_sen+xnames_sub_clscont+xnames_sub_domain+xnames_sub_isscont+xnames_sub_back
    
    xnames_noSimi=xnames_sub_cumu+xnames_sub_act+xnames_sub_sen+xnames_sub_isscont+xnames_sub_back
    xnames_noDev=xnames_sub_clscont+xnames_sub_domain+xnames_sub_isscont+xnames_sub_back
    xnames_noIss=xnames_sub_cumu+xnames_sub_act+xnames_sub_sen+xnames_sub_clscont+xnames_sub_domain

    
    xnames_nocumu=xnames_sub_act+xnames_sub_sen+xnames_sub_clscont+xnames_sub_domain+xnames_sub_isscont+xnames_sub_back
    xnames_noact=xnames_sub_cumu+xnames_sub_sen+xnames_sub_clscont+xnames_sub_domain+xnames_sub_isscont+xnames_sub_back
    xnames_nosen=xnames_sub_cumu+xnames_sub_act+xnames_sub_clscont+xnames_sub_domain+xnames_sub_isscont+xnames_sub_back
    xnames_noclscontent=xnames_sub_cumu+xnames_sub_act+xnames_sub_sen+xnames_sub_domain+xnames_sub_isscont+xnames_sub_back
    xnames_nodomain=xnames_sub_cumu+xnames_sub_act+xnames_sub_sen+xnames_sub_clscont+xnames_sub_isscont+xnames_sub_back
    xnames_noisscontent=xnames_sub_cumu+xnames_sub_act+xnames_sub_sen+xnames_sub_clscont+xnames_sub_domain+xnames_sub_back
    xnames_noback=xnames_sub_cumu+xnames_sub_act+xnames_sub_sen+xnames_sub_clscont+xnames_sub_domain+xnames_sub_isscont

    

    xnames_recgfi=["pro_star","procmt","contributornum","openiss","proclspr","crtclsissnum","openissratio",
                                "rptcmt","rptallcmt","rptpronum","rptalliss","rptiss","rptallpr","rptpr","rptisnew",
                                "ownercmt","ownerallcmt","ownerpronum","owneralliss","owneriss","ownerallpr","ownerpr",
                                "NumOfCode","NumOfUrls","NumOfPics","coleman_liau_index","flesch_reading_ease","flesch_kincaid_grade","automated_readability_index","LengthOfTitle","LengthOfDescription",
                                "gfilabelnum", "buglabelnum", "testlabelnum", "buildlabelnum", "doclabelnum", "enhancelabelnum", "codinglabelnum", "featurelabelnum", "majorlabelnum", "mediumlabelnum", "untriagedlabelnum", "triagedlabelnum", "labelnum",
                                'rpt_reviews_num_all', 'rpt_max_stars_commit', 'rpt_max_stars_issue', 
                                'rpt_max_stars_pull', 'rpt_max_stars_review', 'rpt_gfi_ratio', 'owner_reviews_num_all', 'owner_max_stars_commit', 'owner_max_stars_issue', 'owner_max_stars_pull', 'owner_max_stars_review', 
                                'owner_gfi_ratio', 'owner_gfi_num', 'pro_gfi_ratio', 'pro_gfi_num'
                                ]

    xnames_Stanik=["LengthOfTitle","LengthOfDescription","issuesen",'title',"body"]                              
    dataset_recgfi=[]
    

    path_name="./data/dataset_"
    recgif_path="./data/training_set_recgfi_"



    figind=0
    model_name="LambdaMART"
    ratio1lsts=[]
    ratio3lsts=[]
    ratio5lsts=[]
    ratio10lsts=[]
    firsthitmedianlsts=[]
    for datasetname in ['simcse','codebert','BERTOverflow','roberta']:
        ratio1lst=[]
        ratio3lst=[]
        ratio5lst=[]
        ratio10lst=[]
        firsthitmedianlst=[]
        lst_lens=[]
        for dataset_fold in [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]:
            datasetlst=[]
            for i in range(dataset_fold-1):
                datasetlst.append(pd.read_pickle(path_name+datasetname+"_"+str(i)+".pkl"))
            training_set= pd.concat(datasetlst,axis=0)
            valid_set=pd.read_pickle(path_name+datasetname+"_"+str(dataset_fold-1)+".pkl")
            test_set=pd.read_pickle(path_name+datasetname+"_"+str(dataset_fold)+".pkl")
        
            metrics=get_lamb_metrics(training_set,valid_set,test_set,idname,xnames_LambdaMART)
            ratio1lst.append(metrics[0])
            ratio3lst.append(metrics[1])
            ratio5lst.append(metrics[2])
            ratio10lst.append(metrics[3])
            firsthitmedianlst.append(metrics[4])
        ratio1lsts.append(ratio1lst)
        ratio3lsts.append(ratio3lst)
        ratio5lsts.append(ratio5lst)
        ratio10lsts.append(ratio10lst)
        firsthitmedianlsts.append(firsthitmedianlst)
        print('berts',model_name,np.mean(ratio1lst),np.mean(ratio3lst),np.mean(ratio5lst),np.mean(ratio10lst),np.mean(firsthitmedianlst),np.median(ratio1lst),np.median(ratio3lst),np.median(ratio5lst),np.median(ratio10lst),np.median(firsthitmedianlst))
    
    pos=0
    fig = plt.figure(figind,figsize=(8, 4))
    ax = plt.axes(polar=False)
    figind+=1
    for t in range(5):
        if t==0:
            thelst=ratio1lsts
        elif t==1:
            thelst=ratio3lsts
        elif t==2:
            thelst=ratio5lsts
        elif t==3:
            thelst=ratio10lsts
        elif t==4:
            thelst=firsthitmedianlsts
        list0=thelst[0]
        list1=thelst[1]
        list2=thelst[2]
        list3=thelst[3]
        if t==4:
            ax.twinx()

        bplot=plt.boxplot([list0,list1,list2,list3],widths=0.8,
        positions=[pos,pos+1,pos+2,pos+3], showmeans=True,showfliers=False,patch_artist=True,meanprops = {'markerfacecolor':'black','markeredgecolor':'black', 'markersize':10})
        pos+=5
        [[item.set_color('k') for item in bplot['medians']]]
        colors = ['b','r','g','darkorange']
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

        if t==3:
            plt.tick_params(bottom=False,top=False,left=False,right=False,labelsize=15)
            plt.xticks(fontsize=20)
            plt.ylabel('R@k',{'size':20})
            plt.yticks(fontsize=20)
            ax.spines['right'].set_visible(False)

    plt.tick_params(bottom=False,top=False,left=False,right=False,labelsize=15)
    plt.xlim((-1, 24))
    plt.ylabel('FirstHit',{'size':20})
    plt.xticks([1.5,6.5,11.5,16.5,21.5],["R@1","R@3","R@5","R@10","FH"])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(handles=[bplot["boxes"][0],bplot["boxes"][1],bplot["boxes"][2],bplot["boxes"][3]],labels=['SimCSE (PFIRec)','CodeBERT','BERTOverflow','RoBERTa'],fontsize=13,loc="upper left")
    ax = plt.gca()
    bwith = 2
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    plt.grid(alpha=0.6)
    plt.tight_layout()
    plt.savefig('./figures/berts.png')

   
    datasetname='simcse'
    model_list=["LambdaMART","XGB","Random_forest","Logistic_regression","Stanik","recgfi","random","gfilabel"]
    with open('./data/isstexts.json') as f:
        issuestr = json.load(f)
    issuedata = issuestr['0']
    lst=[]
    for i in range(len(issuedata)):
        lst.append(issuedata[str(i)])
    dfall=pd.DataFrame(lst)
    if "Stanik" in model_list or "recgfi" in model_list:
        for i in range(19):
            dataset_recgfi.append(pd.read_pickle(recgif_path+datasetname+"_"+str(i)+".pkl"))
    ratio1lsts=[]
    ratio3lsts=[]
    ratio5lsts=[]
    ratio10lsts=[]
    firsthitmedianlsts=[]
    for model_name in model_list:
        randseed=0
        if model_name=="recgfi":
            ratio1lst=[]
            ratio3lst=[]
            ratio5lst=[]
            ratio10lst=[]
            firsthitmedianlst=[]
            for dataset_fold in [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]:
                datasetlst=[]
                for i in range(dataset_fold):
                    datasetlst.append(dataset_recgfi[i])
                training_set_recgfi = pd.concat(datasetlst,axis=0)
                test_set=pd.read_pickle(path_name+datasetname+"_"+str(dataset_fold)+".pkl")
                metrics=get_recgfi_metrics(training_set_recgfi,test_set,idname,xnames_recgfi)
                ratio1lst.append(metrics[0])
                ratio3lst.append(metrics[1])
                ratio5lst.append(metrics[2])
                ratio10lst.append(metrics[3])
                firsthitmedianlst.append(metrics[4])
            ratio1lsts.append(ratio1lst)
            ratio3lsts.append(ratio3lst)
            ratio5lsts.append(ratio5lst)
            ratio10lsts.append(ratio10lst)
            firsthitmedianlsts.append(firsthitmedianlst)
        
        elif model_name=="Stanik":
            ratio1lst=[]
            ratio3lst=[]
            ratio5lst=[]
            ratio10lst=[]
            firsthitmedianlst=[]
            for dataset_fold in [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]:
                datasetlst=[]
                for i in range(dataset_fold):
                    datasetlst.append(dataset_recgfi[i])
                training_set_recgfi = pd.concat(datasetlst,axis=0)
                test_set=pd.read_pickle(path_name+datasetname+"_"+str(dataset_fold)+".pkl")
                metrics=get_Stanik_metrics(training_set_recgfi,test_set,idname,xnames_Stanik)
                ratio1lst.append(metrics[0])
                ratio3lst.append(metrics[1])
                ratio5lst.append(metrics[2])
                ratio10lst.append(metrics[3])
                firsthitmedianlst.append(metrics[4])
            ratio1lsts.append(ratio1lst)
            ratio3lsts.append(ratio3lst)
            ratio5lsts.append(ratio5lst)
            ratio10lsts.append(ratio10lst)
            firsthitmedianlsts.append(firsthitmedianlst)


        elif model_name in ["LambdaMART","Logistic_regression","Random_forest","random","XGB","gfilabel"]:
            ratio1lst=[]
            ratio3lst=[]
            ratio5lst=[]
            ratio10lst=[]
            firsthitmedianlst=[]
            for dataset_fold in [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]:
                datasetlst=[]
                for i in range(dataset_fold-1):
                    datasetlst.append(pd.read_pickle(path_name+datasetname+"_"+str(i)+".pkl"))
                training_set = pd.concat(datasetlst,axis=0)
                valid_set=pd.read_pickle(path_name+datasetname+"_"+str(dataset_fold-1)+".pkl")
                test_set=pd.read_pickle(path_name+datasetname+"_"+str(dataset_fold)+".pkl")

                if model_name=="LambdaMART":
                    metrics=get_lamb_metrics(training_set,valid_set,test_set,idname,xnames_LambdaMART)
                elif model_name=="random":
                    metrics=get_random_metrics(test_set,idname)
                elif model_name=="gfilabel":
                    metrics=get_gfilabel_metrics(test_set,idname)
                elif model_name=="XGB":
                    metrics=get_XGB_metrics(training_set,valid_set,test_set,idname,xnames_LambdaMART)
                elif model_name=="Logistic_regression":
                    metrics=get_Logistic_regression_metrics(training_set,valid_set,test_set,idname,xnames_LambdaMART)
                elif model_name=="Random_forest":
                    metrics=get_Random_forest_metrics(training_set,valid_set,test_set,idname,xnames_LambdaMART)
                ratio1lst.append(metrics[0])
                ratio3lst.append(metrics[1])
                ratio5lst.append(metrics[2])
                ratio10lst.append(metrics[3])
                firsthitmedianlst.append(metrics[4])
            ratio1lsts.append(ratio1lst)
            ratio3lsts.append(ratio3lst)
            ratio5lsts.append(ratio5lst)
            ratio10lsts.append(ratio10lst)
            firsthitmedianlsts.append(firsthitmedianlst)

        print('baselines',model_name,np.mean(ratio1lst),np.mean(ratio3lst),np.mean(ratio5lst),np.mean(ratio10lst),np.mean(firsthitmedianlst),np.median(ratio1lst),np.median(ratio3lst),np.median(ratio5lst),np.median(ratio10lst),np.median(firsthitmedianlst))


    pos=0
    fig = plt.figure(figind,figsize=(8, 4))
    ax = plt.axes(polar=False)
    figind+=1
    for t in range(5):
        if t==0:
            thelst=ratio1lsts
        elif t==1:
            thelst=ratio3lsts
        elif t==2:
            thelst=ratio5lsts
        elif t==3:
            thelst=ratio10lsts
        elif t==4:
            thelst=firsthitmedianlsts
        list0=thelst[0]
        list1=thelst[1]
        list2=thelst[2]
        list3=thelst[3]
        if t==4:
            ax.twinx()

        bplot=plt.boxplot([list0,list1,list2,list3],widths=0.8,
        positions=[pos,pos+1,pos+2,pos+3], showmeans=True,showfliers=False,patch_artist=True,meanprops = {'markerfacecolor':'black','markeredgecolor':'black', 'markersize':10})
        pos+=5
        [[item.set_color('k') for item in bplot['medians']]]
        colors = ['b','r','g','darkorange']
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

        if t==3:
            plt.tick_params(bottom=False,top=False,left=False,right=False,labelsize=15)
            plt.xticks(fontsize=20)
            plt.ylabel('R@k',{'size':20})
            plt.yticks(fontsize=20)
            ax.spines['right'].set_visible(False)

    plt.tick_params(bottom=False,top=False,left=False,right=False,labelsize=15)
    plt.xlim((-1, 24))
    plt.ylabel('FirstHit',{'size':20})
    plt.xticks([1.5,6.5,11.5,16.5,21.5],["R@1","R@3","R@5","R@10","FH"])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(handles=[bplot["boxes"][0],bplot["boxes"][1],bplot["boxes"][2],bplot["boxes"][3]],labels=["LambdaMART (PFIRec)","XGBoost","Random_forest","Logistic_regression"],fontsize=13,loc="upper left")
    ax = plt.gca()
    bwith = 2
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    plt.grid(alpha=0.6)
    plt.tight_layout()
    plt.savefig('./figures/MLmodels.png')


    pos=0
    fig = plt.figure(figind,figsize=(8, 4))
    ax = plt.axes(polar=False)
    figind+=1
    for t in range(5):
        if t==0:
            thelst=ratio1lsts
        elif t==1:
            thelst=ratio3lsts
        elif t==2:
            thelst=ratio5lsts
        elif t==3:
            thelst=ratio10lsts
        elif t==4:
            thelst=firsthitmedianlsts
        list0=thelst[0]
        list4=thelst[4]
        list5=thelst[5]
        list6=thelst[6]
        list7=thelst[7]
        if t==4:
            ax.twinx()

        bplot=plt.boxplot([list0,list4,list5,list6,list7],widths=0.8,
        positions=[pos,pos+1,pos+2,pos+3,pos+4], showmeans=True,showfliers=False,patch_artist=True,meanprops = {'markerfacecolor':'black','markeredgecolor':'black', 'markersize':10})
        pos+=6
        [[item.set_color('k') for item in bplot['medians']]]
        colors = ['b','r','g','darkorange','purple']
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

        if t==3:
            plt.tick_params(bottom=False,top=False,left=False,right=False,labelsize=15)
            plt.xticks(fontsize=20)
            plt.ylabel('R@k',{'size':20})
            plt.yticks(fontsize=20)
            ax.spines['right'].set_visible(False)

    plt.tick_params(bottom=False,top=False,left=False,right=False,labelsize=15)
    plt.xlim((-1, 29))
    plt.ylabel('FirstHit',{'size':20})
    plt.xticks([2,8,14,20,26],["R@1","R@3","R@5","R@10","FH"])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(handles=[bplot["boxes"][0],bplot["boxes"][1],bplot["boxes"][2],bplot["boxes"][3],bplot["boxes"][4]],labels=["PFIRec","Stanik et al.","GFI-bot","Random","GFIRandom"],fontsize=13,loc="upper left")
    ax = plt.gca()
    bwith = 2
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    plt.grid(alpha=0.6)
    plt.tight_layout()
    plt.savefig('./figures/baselines.png')

    datasetname='simcse'
    model_list=["xnames_noclscontent","xnames_nodomain","xnames_nogener","xnames_noact","xnames_nosen","xnames_noisscontent","xnames_noback","crosspro","LambdaMART"]
    ratio1lsts=[]
    ratio3lsts=[]
    ratio5lsts=[]
    ratio10lsts=[]
    firsthitmedianlsts=[]
    for model_name in model_list:
        ratio1lst=[]
        ratio3lst=[]
        ratio5lst=[]
        ratio10lst=[]
        firsthitmedianlst=[]
        for dataset_fold in [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]:
            datasetlst=[]
            for i in range(dataset_fold-1):
                datasetlst.append(pd.read_pickle(path_name+datasetname+"_"+str(i)+".pkl"))
            training_set = pd.concat(datasetlst,axis=0)
            valid_set=pd.read_pickle(path_name+datasetname+"_"+str(dataset_fold-1)+".pkl")
            test_set=pd.read_pickle(path_name+datasetname+"_"+str(dataset_fold)+".pkl")

            if model_name=="LambdaMART":
                metrics=get_lamb_metrics(training_set,valid_set,test_set,idname,xnames_LambdaMART)
            elif model_name=="xnames_noSimi":
                metrics=get_lamb_metrics(training_set,valid_set,test_set,idname,xnames_noSimi)
            elif model_name=="xnames_noDev":
                metrics=get_lamb_metrics(training_set,valid_set,test_set,idname,xnames_noDev)
            elif model_name=="xnames_noIss":
                metrics=get_lamb_metrics(training_set,valid_set,test_set,idname,xnames_noIss)

            elif model_name=="xnames_nogener":
                metrics=get_lamb_metrics(training_set,valid_set,test_set,idname,xnames_nocumu)
            elif model_name=="xnames_noact":
                metrics=get_lamb_metrics(training_set,valid_set,test_set,idname,xnames_noact)
            elif model_name=="xnames_nosen":
                metrics=get_lamb_metrics(training_set,valid_set,test_set,idname,xnames_nosen)
            elif model_name=="xnames_noclscontent":
                metrics=get_lamb_metrics(training_set,valid_set,test_set,idname,xnames_noclscontent)
            elif model_name=="xnames_nodomain":
                metrics=get_lamb_metrics(training_set,valid_set,test_set,idname,xnames_nodomain)
            elif model_name=="xnames_noisscontent":
                metrics=get_lamb_metrics(training_set,valid_set,test_set,idname,xnames_noisscontent)
            elif model_name=="xnames_noback":
                metrics=get_lamb_metrics(training_set,valid_set,test_set,idname,xnames_noback)
            
            elif model_name=="crosspro":
                training_set['proname'] = training_set.apply(lambda row: get_proname(row['owner'], row['name']), axis=1)
                valid_set['proname'] = valid_set.apply(lambda row: get_proname(row['owner'], row['name']), axis=1)
                test_set['proname'] = test_set.apply(lambda row: get_proname(row['owner'], row['name']), axis=1)
                trainpro=list(training_set['proname'].values)
                validpro=list(valid_set['proname'].values)
                testpro=list(test_set['proname'].values)
                pronamelst=trainpro+validpro+testpro
                pronamelst=list(set(pronamelst))
                pronamelst.sort()

                metric0=0
                metric1=0
                metric2=0
                metric3=0
                metric4=0
                addcount=0
                for fold in range(10):
                    p_train_split1=int((fold/10)*len(pronamelst))
                    p_train_split2=int((fold/10+0.1)*len(pronamelst))

                    trainvali_pro=pronamelst[:p_train_split1]+pronamelst[p_train_split2:]
                    test_pro=pronamelst[p_train_split1:p_train_split2]


                    train_subdata=training_set[training_set['proname'].isin(trainvali_pro)]
                    valid_subdata=valid_set[valid_set['proname'].isin(trainvali_pro)]
                    test_subdata=test_set[test_set['proname'].isin(test_pro)]

                    crometrics=get_lamb_metrics(train_subdata,valid_subdata,test_subdata,idname,xnames_LambdaMART)
                    print(crometrics)
                    if not isinstance(crometrics[0], (int, float)):
                        continue
                    else:
                        metric0+=crometrics[0]
                        metric1+=crometrics[1]
                        metric2+=crometrics[2]
                        metric3+=crometrics[3]
                        metric4+=crometrics[4]
                        addcount+=1
                if addcount==0:
                    metrics=None
                else:
                    metrics=[metric0/addcount,metric1/addcount,metric2/addcount,metric3/addcount,metric4/addcount]

            if metrics is None:
                continue
            ratio1lst.append(metrics[0])
            ratio3lst.append(metrics[1])
            ratio5lst.append(metrics[2])
            ratio10lst.append(metrics[3])
            firsthitmedianlst.append(metrics[4])
        ratio1lsts.append(ratio1lst)
        ratio3lsts.append(ratio3lst)
        ratio5lsts.append(ratio5lst)
        ratio10lsts.append(ratio10lst)
        firsthitmedianlsts.append(firsthitmedianlst)

    for i in range(len(model_list)):
        print(model_list[i],np.mean(ratio1lsts[i]),np.mean(ratio3lsts[i]),np.mean(ratio5lsts[i]),np.mean(ratio10lsts[i]),np.mean(firsthitmedianlsts[i]),np.median(ratio1lsts[i]),np.median(ratio3lsts[i]),np.median(ratio5lsts[i]),np.median(ratio10lsts[i]),np.median(firsthitmedianlsts[i]))
    print("finish")