lis=[[3,3,4,4,0,3,4,3,0,0,4,4],['img_1',"img_1",'img_1','img_1','img_2','img_2','img_2','img_2','img_2','img_2','img_2','img_2'],['img_1',"img_1",'img_1','img_1','img_2','img_2','img_2','img_2','img_2','img_2','img_2','img_2']]
#加一个映射函数
def gleason_to_up_score(gleason_1,gleason_2):
    if gleason_1==3 and gleason_2==3:
        return 1
    if gleason_1==3 and gleason_2==4:
        return 2
    if gleason_1==4 and gleason_2==3:
        return 3
    if gleason_1==4 and gleason_2==4:
        return 4
    if gleason_1==3 and gleason_2==5:
        return 4
    if gleason_1==5 and gleason_2==3:
        return 4
    if gleason_1==4 and gleason_2==5:
        return 5
    if gleason_1==5 and gleason_2==4:
        return 5
    if gleason_1==5 and gleason_2==5:
        return 5
    return 0
#加一个阈值函数
def yuzhi(gleason_1,gleason_2,beishu):
    if gleason_1 / (gleason_2+0.0001)>beishu:
        return gleason_1,gleason_1
    return gleason_1,gleason_2




# img (0, 'img_1')
list2 = []
img_tag = None
out_list = []
p_list = [0, 0, 0, 0, 0, 0]
i = 0  # i记录当前第几张图片 s记录一共有多少张patch p0,1, 3，4，5
list2 = []
img_tag = None
out_list = []
p_list = [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0]]
i = 0  # i记录当前第几张图片 s记录一共有多少张patch p0,1, 3，4，5
for s, img in zip(enumerate(lis[0]), enumerate(lis[2])):
    if img[1] != img_tag:
        p_list2=sorted(p_list, key=(lambda x: x[1]), reverse=True)
        gleason1, gleason2 = p_list2[0][0], p_list2[1][0]
        # gleason1,gleason2=yuzhi(gleason1,gleason2,1)
        up_score = gleason_to_up_score(gleason1, gleason2)
        out_list.append([img_tag, gleason1, gleason2, up_score])
        i = i + 1
        img_tag = img[1]
        p_list = [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0]]
    if s[1] == 0:
        p_list[0][1] = p_list[0][1] + 1
    if s[1] == 1:
        p_list[1][1] = p_list[1][1] + 1
    if s[1] == 3:
        p_list[3][1] = p_list[3][1] + 1
    if s[1] == 4:
        p_list[4][1] = p_list[4][1] + 1
    if s[1] == 5:
        p_list[5][1] = p_list[5][1] + 1
p_list2=sorted(p_list, key=(lambda x: x[1]), reverse=True)
gleason1, gleason2 = p_list2[0][0], p_list2[1][0]
# gleason1,gleason2=yuzhi(gleason1,gleason2,1)
up_score = gleason_to_up_score(gleason1, gleason2)
out_list.append([img_tag, gleason1, gleason2, up_score])
print(out_list)