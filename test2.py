import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

img1=cv2.imread('C:\\Users\dell\Desktop\RXX-2020-CVA\RXX-2020-CVA\Testing dataset\TianJing-SPOT/09BMP.bmp')
img2=cv2.imread('C:\\Users\dell\Desktop\RXX-2020-CVA\RXX-2020-CVA\Testing dataset\TianJing-SPOT/10BMP.bmp')

def Find_Threshold(delta):#OTSU寻找阈值
    # 求灰度方差最大的那个数
    #主要是利用方差来进行计算

    val=np.zeros([256])
    for th in range(256):
        loc1=delta>th
        # loc1 为true 或false 的矩阵
        loc2=delta<=th
        '''delta[loc1]=255
        delta[loc2]=0'''
        #delta[loc1] 将为True 的矩阵元素提取出来为1维
        if delta[loc1].size==0:
            mu1=0
            omega1=0
        else:
            mu1=np.mean(delta[loc1])
            #平均值
            omega1=delta[loc1].size/delta.size
            # 所占比例
        if delta[loc2].size==0:
            mu2=0
            omega2=0
        else:
            mu2=np.mean(delta[loc2])
            omega2=delta[loc2].size/delta.size
        
        #最终推导的公式为  g=w0*w1*()
        # 这里使用了最后的推导公式原文地址https://blog.csdn.net/guoyk1990/article/details/7606032?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.nonecase&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.nonecase
        val[th]=omega1*omega2*np.power((mu1-mu2),2)

    #print("val=",val.shape)
    plt.figure()
    # val 存放的是类间方差 取其最大
    loc=np.where(val==np.max(val))
    #x=np.arange(0,256,1)
    #x=x.reshape([1,256])
    plt.plot(val)
    plt.ylabel("Var")
    plt.xlabel("Threshold")
    plt.grid("on")

    print("\nThe best OTSU Threshold: ",loc[0])
    return loc[0]

def CD_diff(img1,img2):#影像差值法
    delta=cv2.subtract(img1,img2)
    #delta=np.abs(delta)
    #delta.min()
    sh=delta.shape
    th=Find_Threshold(delta)
    # 设为灰度值 再进行运算
    delta = cv2.cvtColor(delta, cv2.COLOR_BGR2GRAY)
    #print(delta.min())
    if np.size(th)>1:
        th=th[0]
    for i1 in range(sh[0]):
        for i2 in range(sh[1]):
            if delta[i1][i2]<=th:
                delta[i1][i2]=255
            else:
                delta[i1][i2]=0
    return delta


def divede(img1,img2):
    delta=cv2.divide(img2,img1)
    #delta=np.abs(delta)
    #delta.min()
    (mean1,stddv1)=cv2.meanStdDev(delta)
    gray2 = cv2.cvtColor(delta, cv2.COLOR_BGR2GRAY)
    # 比值法的阈值确定常采用均值
    ret2,thread2=cv2.threshold(gray2,mean1[0]+1,255,cv2.THRESH_BINARY)
    return thread2

def rcva(img1,img2): 
    b,g,red=cv2.split(img1)
    b1,g1,red1=cv2.split(img2)

    (row,cloumn)=b.shape
    #上下左右 左上 右上 左下 右下的坐标变化
    x=[-1,1,0,0,-1,-1,1,1]
    y=[0,0,-1,1,-1,1,-1,1]
    
    #保存变化后的差异图
    img_a=np.zeros((row,cloumn))
    img_b=np.zeros((row,cloumn))
    print("正在计算")
    for i in range(row):
        for j in range(cloumn):
            if i!=0 and j!=0 and i!=row and j!=cloumn:
                r=i
                c=j
                res1=1000000.0
                res2=1000000.0
                res3=1000000.0
                for k in range(8):
                    if(0<r and r<row and c>0 and c<cloumn):
                        kk1=(math.pow(abs(int(b[i,j])-int(b1[r,c])),2))
                        if(kk1<res1):
                            res1=kk1
                        kk2=(math.pow(abs(int(g[i,j])-int(g1[r,c])),2))
                        if(kk2<res2):
                            res2=kk2
                        kk3=(math.pow(abs(int(red[i,j])-int(red1[r,c])),2))
                        if(kk3<res3):
                            res3=kk3
                    r=i
                    c=j
                    r+=x[k]
                    c+=y[k]
                img_a[i][j]=math.sqrt(res1+res2+res3)
                
        # 考虑四个边角 和四个边界
            else:
                img_a[i,j]=abs(int(b[i,j])-int(b1[i,j]))
    
    print("正在计算中请稍等")
    
    for i in range(row):
        for j in range(cloumn):
            if i!=0 and j!=0 and i!=row and j!=cloumn:
                r=i
                c=j
                res1=1000000.0
                res2=1000000.0
                res3=1000000.0
                for k in range(8):
                    if(0<r and r<row and c>0 and c<cloumn):
                        kk1=(math.pow(abs(int(b[i,j])-int(b1[r,c])),2))
                        if(kk1<res1):
                            res1=kk1
                        kk2=(math.pow(abs(int(g[i,j])-int(g1[r,c])),2))
                        if(kk2<res2):
                            res2=kk2
                        kk3=(math.pow(abs(int(red[i,j])-int(red1[r,c])),2))
                        if(kk3<res3):
                            res3=kk3
                    r=i
                    c=j
                    r+=x[k]
                    c+=y[k]
                img_b[i][j]=math.sqrt(res1+res2+res3)
        # 考虑四个边角 和四个边界
            else:
                img_b[i,j]=abs(int(b1[i,j])-int(b[i,j]))

    img_b_change=np.zeros((row,cloumn))
    for i in range(row):
        for j in range(cloumn):
            if img_a[i,j]>img_b[i,j]:
                img_b_change[i,j]=img_b[i,j]
            else:
                img_b_change[i,j]=img_a[i,j]

    print("计算完成正在二值化")
    th=Find_Threshold(img_b_change)
    
    for i1 in range(row):
        for i2 in range(cloumn):
            if img_b_change[i1][i2]<=th:
                img_b_change[i1][i2]=0
            else:
                img_b_change[i1][i2]=255
    return img_b_change


def cva(img1,img2):
    b,g,r=cv2.split(img1)
    b1,g1,r1=cv2.split(img2)

    (row,col)=b1.shape

    d_1=cv2.subtract(b,b1)
    
    d_2=cv2.subtract(g,g1)
    d_3=cv2.subtract(r,r1)

    d_1=cv2.pow(d_1,2)
    d_2=cv2.pow(d_2,2)
    d_3=cv2.pow(d_3,2)

    d_sum_2=np.add(d_1,d_2,d_3)
    #d_sum_2 = cv2.cvtColor(d_sum_2, cv2.COLOR_BGR2GRAY)
    
    for i in range(row):
        for j in range(col):
            d_sum_2[i,j]=math.sqrt(d_sum_2[i,j])
    delta=d_sum_2
    
    
    th=Find_Threshold(delta)
   
    if np.size(th)>1:
        th=th[0]
    for i1 in range(row):
        for i2 in range(col):
            if delta[i1][i2]<=th:
                delta[i1][i2]=0
            else:
                delta[i1][i2]=255
    return delta


# rcva 方法
#def R_Cva():
    

# 主成向量分析法
# 这个是二维图像的pca方法
def Img_PCA(delta):
    #奇异值分解用于压缩数据找主成分
    # 此方法好想只进行了SVD 分解
    U,S,V=np.linalg.svd(delta)
    SS=np.zeros(U.shape)
    print(SS.shape)
    
    # S也是二维元祖，只不过第二个数为空 （7，）
    for i in range(S.shape[0]):
        SS[i][i]=S[i]

    def Pick_k(S):
        sval=np.sum(S)
        for i in range(S.shape[0]):
            # 去大于70% 的主成分
            if np.sum(S[:i])>=0.7*sval:
                break
        return i+1

    k=Pick_k(S)
    print("\nNumber of vectors to reserve: k= ",k)
    Uk=U[:,0:k]
    Sk=SS[0:k,0:k]
    Vk=V[0:k,:]
    # 点积运算
    im=np.dot(np.dot(Uk,Sk),Vk)
   # im=np.dot(im,delta)
    return im


#得到一幅图像的灰度值概率密度，是一个大小256的数组         
def myGetValueProbability(img1):
    # img1 转进来应该是一张灰度图
    #img1 是全黑的   全为0
    #cv2.imshow('diff',img1)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    (row,col)=img1.shape
    p=np.array([0.0 for i in range(256)])
    d0=np.array([0.0 for i in range(256)])
    for i in range(row):
        for j in range(col):
            p[int(img1[i,j])]+=1
    #统计概率
    for i in range(256):
        pp=p[i]
        d0[i]=pp/(row*col)
    
    #返回概率密度
    return d0


def myGetEntropy(img1):
    temp_img=img1
    (row,col)=img1.shape
    entropy=0
    d=myGetValueProbability(temp_img)
    #print(d[:10])
    #dm 可能的最小概率
    dm=1.0/(row*col)/10.0
    
    for i in range(len(d)):
        if d[i]>dm and d[i]<=1:
            l=d[i]
            entropy+=(-l*math.log(l))
    
    return entropy



# 图像的定权 根据熵来更新图片
# isED 决定定权之后是计算欧式距离还是直接线性加权
def MyEntropyWeights(img1,img2,isED):
    # img1 和img2 都是单通道的
    (row,col)=img1.shape
    entropy1 = 0.0
    entropy2 = 0.0
    maritx_temp=np.array([[0.0]*col for i in range(row)])
    # 定义权
    w1=0.0
    w2=0.0
    # 计算熵和权
    entropy1 = myGetEntropy(img1)
    entropy2 =myGetEntropy(img2)
 
    #采用反熵定权，因为数值大小不一样，熵越小权重越大
    w2 = entropy1 / (entropy1 + entropy2)
    w1 = 1 - w2
    
    #基于熵的加权平均
    s1=0.0
    s2=0.0
    tempDN=0.0
    for i in  range(row):
        for j in range(col):
            s1=img1[i,j]
            s2=img2[i,j]
            if isED:
                tempDN=(w1*s1)*(w1*s1) + (w2*s2)*(w2*s2)
                maritx_temp[i,j]=math.sqrt(tempDN)
            else:
                maritx_temp[i,j]=w1*s1 + w2*s2
            
    
    # 再进行一次正则化
    #image = (maritx_temp- np.min(maritx_temp)) / (np.max(maritx_temp) - np.min(maritx_temp))
    image = 255*(maritx_temp-np.min(maritx_temp))/(np.max(maritx_temp)-np.min(maritx_temp))

    th=Find_Threshold(image)
    
    for i1 in range(row):
        for i2 in range(col):
            if image[i1][i2]>=th:
                image[i1][i2]=255
            else:
                image[i1][i2]=0
            
    cv2.imshow('AFS算法',image)
    #cv2.imwrite("..\RXX-2020-CVA\Testing dataset\B-landslideAerialImage/AFS.bmp",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return image

                    
    
    
            
# AFS 代码实现
def AFS(img1,img2):
    (row1,col1,dim1)=img1.shape
    (row2,col2,dim2)=img2.shape
    
    if dim1!=dim2:
        print("维度不同，无法比较")
    temp_img1=img1
    temp_img2=img2
    #创建两个二维矩阵来存放数据
    matrix_CVA = np.array([[0.0]*col1 for i in range(row1)])
    matrix_SAM = np.array([[0.0]*col1 for i in range(row1)])
    
    SAD=0.0  # SAD 存放光谱角
    tagSum=0.0
    tagEum=0.0
    s1=0.0
    s2=0.0
    Ax = 0.0
    Ay1 = 0.0
    Ay2 = 0.0
    Ay3 =0.0
    A=0.0 #A存放余弦值
    # 计算自适应AFS 差值（SAM+CVA）
    for i in range(row1):
        for j in range(col1):
            tagSum=0.0
            Ax=0.0
            Ay1=0.0
            Ay2=0.0
            Ay3=0.0
            A =0.0
            SAD = 0.0            
            for k in range(dim1):
                kk1=int(temp_img1[i,j,k])
                kk2=int(temp_img2[i,j,k]) #向量内积
                Ax+=kk1*kk2
                #print(temp_img1[i,j,k],temp_img2[i,j,k],temp_img1[i,j,k]*temp_img2[i,j,k],i,j,k,Ax)
                Ay1+=int(temp_img1[i,j,k])*int(temp_img1[i,j,k])
                Ay2+=int(temp_img2[i,j,k])*int(temp_img2[i,j,k])
                tagEum=0.0
                s1=int(temp_img1[i,j,k])
                s2=int(temp_img2[i,j,k])
                tagEum=abs(s1-s2)
                tagSum+=tagEum*tagEum
            # 运算完是三个通道的结果
            #tagSum 保存的是差值的平方和
           
            Ay1=math.sqrt(Ay1)
            Ay2=math.sqrt(Ay2)
            Ay3=Ay1*Ay2

            if Ay1==0 or Ay2==0:
                SAD=0
            elif Ax==0: # 内积为0
                SAD=90
            else:
                A=Ax/Ay3
                #print(A)
                AA=abs(A-1)
                if AA<1e-7:
                    SAD=0
                else:
                    SAD=math.acos(A)*180.0/3.1415926
            tagSum=math.sqrt(tagSum)
            
            SAD=2.8*SAD
            matrix_CVA[i,j]=tagSum
            matrix_SAM[i,j]=SAD

            
            
            #归一化 熵的函数后面添加
    matrix_CVA = 255*(matrix_CVA- np.min(matrix_CVA)) / (np.max(matrix_CVA) - np.min(matrix_CVA))            
    matrix_SAM = 255*(matrix_SAM- np.min(matrix_SAM)) / (np.max(matrix_SAM) - np.min(matrix_SAM))

    MyEntropyWeights(matrix_SAM,matrix_CVA,0)  
           
                

                
def getDifferenceImageVectorEDSAD(img1,img2,isHFV):
    (row1,col1,dim1)=img1.shape
    (row2,col2,dim2)=img2.shape
    
    if dim1!=dim2:
        print("维度不同，无法比较")
    temp_img1=img1
    temp_img2=img2
    #创建三个个二维矩阵来存放数据
    Dst=np.array([[0.0]*col1 for i in range(row1)])
    matrix_CVA = np.array([[0.0]*col1 for i in range(row1)])
    matrix_SAM = np.array([[0.0]*col1 for i in range(row1)])
    
    SAD=0.0  # SAD 存放光谱角
    tagSum=0.0
    tagEum=0.0
    s1=0.0
    s2=0.0
    Ax = 0.0
    Ay1 = 0.0
    Ay2 = 0.0
    Ay3 =0.0
    A=0.0 #A存放余弦值
    maxCVA=-10000.0
    maxSAM = -10000.0 # 记录变化矢量和光谱角的最大值
    k=1.0 # 比例系数初值为1
    # 计算自适应AFS 差值（SAM+CVA）
    for i in range(row1):
        for j in range(col1):
            tagSum=0.0
            Ax=0.0
            Ay1=0.0
            Ay2=0.0
            Ay3=0.0
            A =0.0
            SAD = 0.0            
            for k in range(dim1):
                kk1=int(temp_img1[i,j,k])
                kk2=int(temp_img2[i,j,k]) #向量内积
                Ax+=kk1*kk2
                #print(temp_img1[i,j,k],temp_img2[i,j,k],temp_img1[i,j,k]*temp_img2[i,j,k],i,j,k,Ax)
                Ay1+=int(temp_img1[i,j,k])*int(temp_img1[i,j,k])
                Ay2+=int(temp_img2[i,j,k])*int(temp_img2[i,j,k])
                tagEum=0.0
                s1=int(temp_img1[i,j,k])
                s2=int(temp_img2[i,j,k])
                tagEum=abs(s1-s2)
                tagSum+=tagEum*tagEum
            # 运算完是三个通道的结果
            #tagSum 保存的是差值的平方和
           
            Ay1=math.sqrt(Ay1)
            Ay2=math.sqrt(Ay2)
            Ay3=Ay1*Ay2

            if Ay1==0 or Ay2==0:
                SAD=0
            elif Ax==0: # 内积为0
                SAD=90
            else:
                A=Ax/Ay3
                #print(A)
                AA=abs(A-1)
                if AA<1e-7:
                    SAD=0
                else:
                    SAD=math.acos(A)*180.0/3.1415926
            tagSum=math.sqrt(tagSum)
            maxCVA=max(maxCVA,tagSum)
            maxSAM=max(maxSAM,SAD)
            
            #SAD=2.8*SAD
            matrix_CVA[i,j]=tagSum
            matrix_SAM[i,j]=SAD

    if(isHFV):
        k = 255.0 / 90.0
    else:
        if maxSAM!=0:
            k=maxCVA/maxSAM # GCVA
        else:
            k=1
    for i in range(row1):
        for j in range(col1):
            tagSum=0.0
            SAD=0.0
            SAD=k* matrix_SAM[i,j]
            tagSum+=matrix_CVA[i,j]*matrix_CVA[i,j]
            tagSum+=SAD*SAD
            tagSum=math.sqrt(tagSum) #计算GCVA的模长
            Dst[i,j]=tagSum
            

            
            #归一化 熵的函数后面添加
    Dst = 255*(Dst- np.min(Dst)) / (np.max(Dst) - np.min(Dst))
    th=Find_Threshold(Dst)
    
    for i1 in range(row1):
        for i2 in range(col1):
            if Dst[i1][i2]>=th:
                Dst[i1][i2]=255
            else:
                Dst[i1][i2]=0
    cv2.imshow('HFV 算法',Dst)
    cv2.imwrite("..\RXX-2020-CVA\Testing dataset\B-landslideAerialImage/HFV.bmp",Dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return Dst


def evevalue_it(img1,img2):
    # img1 the true picture
    #img2 the detected pictrue
    # 图像检测评价
    #两个算法通过对比黑色表示变化部分
    #不一定黑色就是变化部分

    x00=0
    x10=0
    x01=0
    x11=0
    (row,cloumn)=img1.shape
    for i in range(row):
        for j in range(cloumn):
            if img1[i][j]==0 and img2[i][j]==0:
                x00+=1
            if img1[i][j]==255 and img2[i][j]==255:
                x11+=1
            if img1[i][j]==255 and img2[i][j]==0:
                x10+=1
            if img1[i][j]==0 and img2[i][j]==255:
                x01+=1
    
    print(x00,x11,x10,x01)
    TRC=x00+x10
    print(TRC)
    TRU=x01+x11
    print(TRU)
    TDC=x00+x01
    print(TDC)
    TDU=x10+x11
    T=TRC+TDC
    #总错误率
    
    total_erro=(x01+x10)/T
    #总误错率
    fake_erro=x01/TRU
    #误漏率
    dn_find_erro=x10/TRC
    #整体精度
    total_accucy=(x00+x11)/T
    print("总错误率为%f" %total_erro+"总误错率为%f" %fake_erro+"漏错率为%f" %dn_find_erro,"整体精度为%f" %total_accucy,sep=";")
