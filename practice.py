def reversal(arr):
    i=0
    j=len(arr)-1
    while(i<j):
        arr[i],arr[j]=arr[j],arr[i]
    return arr
print(reversal([1,2,3]))