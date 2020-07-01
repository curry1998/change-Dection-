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

