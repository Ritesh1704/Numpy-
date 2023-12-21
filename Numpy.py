#!/usr/bin/env python
# coding: utf-8

# # Python NumPy

# - Ancestor of NumPy(Numerical Python) is Numeric
# - In `2005`, Travis Oilphant developed NumPy
# - Useful library for scientific computing
# - NumPy does a real good job on linear algebra operations, can be used as an alternate to `MATLAB`
# - It is a very useful library to perform mathematical and statistical operations in Python.
# - It provides a high-performance multidimensional array object
# - NumPy is memory efficient

# ### Why do use NumPy?

# In[1]:


list1 = [1,2,3]
list2 = [2,4,6]


# In[2]:


print(list1)


# In[3]:


print(list2)


# - Multiply the given two lists to generate the output as
# 
# `out_list = [2,8,18]` #elementwise multiplication

# ### Multiple ways to achieve this

# #### Using `for` loop

# In[4]:


list1 = [1,2,3]
list2 = [2,4,6]

result = []

for i in range(len(list1)):
    result.append(list1[i]*list2[i])

print(result)


# #### Using `list comprehension`

# In[5]:


list1 = [1,2,3]
list2 = [2,4,6]

result = [list1[i]*list2[i] for i in range(len(list1))]

print(result)


# #### Using `zip function`

# In[6]:


list1 = [1,2,3]
list2 = [2,4,6]

result = [x*y for x,y, in zip(list1,list2)]

print(result)


# #### Using `map()` and `lambda expresssion`

# In[1]:


list1 = [1,2,3]
list2 = [2,4,6]

result =list(map(lambda x,y: x*y, list1, list2))

print(result)


# In[8]:


import numpy as np #importing library
import os


# `np.array` => create an array from list or any other object

# In[9]:


list1 * list2


# **Elementwise multiplication of two list - vectorized operations not possible in Python using lists**

# In[ ]:


arr1 = np.array(list1)
print(arr1)
print(type(arr1))


# In[ ]:


list1


# In[ ]:


arr2 = np.array(list2)
print(arr2)
print(type(arr2))


# In[ ]:


out_arr = arr1*arr2
print(out_arr)
print(type(out_arr))


# #### Convert the output back from array to list

# In[ ]:


out_list = out_arr.tolist()


# In[10]:


print(out_list)


# `.tolist()` # converts it to list

# In[ ]:


os.listdir() #get the list of files in current working directory


# In[ ]:


os.getcwd() #get current working directory


# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ### Array Creation and Initialization

# - Creating an array means `declaring an array in the memory` and it **can be empty**
# - Initialize means assigning values to an array, it **won't be empty**

# ### 1- dimensional array

# In[11]:


my_first_Array = np.array([7,4,50,90,6])
print('1-D Array', my_first_Array)


# In[12]:


print(type(my_first_Array))


# `nd` means `n-dimensional`

# #### NumPy arrays are homogeneous and can contain object of only one type

# In[13]:


list_3 = [7,3,8,90,"APC", 2+3j, 7.5]


# In[14]:


print(list_3)


# In[15]:


arr_3=np.array(list_3)


# In[16]:


print(arr_3)


# - Observation: In case, array is initialized with hetereogeneous data types, it converts it into string/character data type as default. This is known as implicit typecasting

# ### Typical NumPy basic functions

# `Inspection Functions`

# - ndim: number of dimensions
# 
# - shape:returns a tuple with each index having the number of corresponding elements
# 
# - size: it counts the no. of elements along a given axis, **by default it will count total no. of elements in array**
# 
# - dtype: data type of array elements
# 
# - itemsize: byte size of **each array element**
# 
# - nbytes: total size of array and it is equal to `itemsize X size`

# In[18]:


arr_1d =np.array([1,10,12,16,30])
print(arr_1d)


# In[19]:


type(arr_1d)


# In[25]:


print('Dimension of the array:', arr_1d.ndim)
print('Shape of the array:', arr_1d.shape)
print('Size of the array:', arr_1d.size)
print('Datatype of the dtype:', arr_1d.dtype)
print('Itemsize of the array:', arr_1d.itemsize)
print('Total size of the array:', arr_1d.nbytes)


# ### 2- dimensional array

# ![image.png](attachment:image.png)

# In[26]:


arr_2d = np.array([[5.2,3.0,4.5],
                   [9.1,0.1,0.3]])


# In[27]:


print(arr_2d)


# In[28]:


type(arr_2d)


# In[29]:


print('Dimension of the array:', arr_2d.ndim)
print('Shape of the array:', arr_2d.shape)
print('Size of the array:', arr_2d.size)
print('Datatype of the dtype:', arr_2d.dtype)
print('Itemsize of the array:', arr_2d.itemsize)
print('Total size of the array:', arr_2d.nbytes)


# In[30]:


arr_2d_2 = np.array([[5.2,3.0,4.5],
                   [9.1,0.1,0.3],
                  [1,10,20]])


# In[31]:


arr_2d_2.ndim


# In[32]:


arr_2d_2.dtype


# #### Assign the `dtype` as per your feasible choice 

# In[33]:


arr_2d = np.array([[5.2,3.0,4.5],
                   [9.1,0.1,0.3]],dtype='float32')


# In[34]:


arr_2d.dtype


# In[35]:


arr_2d.nbytes


# ### 3- dimensional Array

# ![image.png](attachment:image.png)
# `credit: to the infographics creator`

# In[36]:


arr_3d = np.array([
    [[10,11,12],[13,14,15], [16,17,18]],    #first layer-2D
    [[20,21,22],[23,24,25], [26,27,28]],    #second layer-2D
    [[30,31,32],[33,34,35], [36,37,38]]    #third layer-2D
])


# In[38]:


print(arr_3d)


# In[39]:


print('Dimension of the array:', arr_3d.ndim)
print('Shape of the array:', arr_3d.shape)
print('Size of the array:', arr_3d.size)
print('Data type of the array:', arr_3d.dtype)
print('Itemsize of the array:', arr_3d.itemsize)
print('Total size of of the array:', arr_3d.nbytes)


# In[40]:


arr_3d = np.array([
    [[10,11,12],[13,14,15], [16,17,18]],    #first layer-2D
    [[20,21,22],[23,24,25], [26,27,28]],    #second layer-2D
    [[30,31,32],[33,34,35], [36,37,38]],    #third layer-2D
    [[40,41,42],[43,44,45], [46,47,48]]     #fourth layer -2D
])


# In[41]:


print('Dimension of the array:', arr_3d.ndim)
print('Shape of the array:', arr_3d.shape)
print('Size of the array:', arr_3d.size)
print('Data type of the array:', arr_3d.dtype)
print('Itemsize of the array:', arr_3d.itemsize)
print('Total size of of the array:', arr_3d.nbytes)


# ### H/W Create a simple 4-D array 

# #### Initialize all the elements of the array of your choice with `0`

# `np.zeros`

# In[47]:


arr_zero = np.zeros((3,3,3), dtype='int32')
print(arr_zero)


# #### Initialize all the elements of the array of your choice with `1`

# `np.ones`

# In[48]:


arr_ones = np.ones((3,3,3), dtype='int32')
print(arr_ones)


# **shape(i,j,k)**

# `i`: `number of layers`
# 
# `j`: `number of rows`
# 
# `k`: `number of columns`

# ![image.png](attachment:image.png)

# #### Initialize all the elements with any `fixed number`

# `np.full`

# In[49]:


arr_full = np.full((3,3,3), 7)
print(arr_full)


# In[50]:


arr_full = np.full((3,3,3), "APC")
print(arr_full)


# #### Filling RANDOM numbers in an array of dimension X x Y

# `np.random.random()`

# In[55]:


arr_random = np.random.random((3,3,3))
print(arr_random)


# ![image.png](attachment:image.png)

# ### Creating normal distribution (random data) having `mean` and `std. dev.` of your choice

# ![image.png](attachment:image.png)

# `np.random.normal()`

# In[56]:


arr_random_normal = np.random.normal(10,2,(3,3,3))


# ![image.png](attachment:image.png)

# In[57]:


print(arr_random_normal)


# In[58]:


arr_random_normal.mean()


# In[59]:


arr_random_normal.std()


# ### `Reading assignment:  Normal distribution vs Standard normal distirbution`

# ![image.png](attachment:image.png)

# #### Print an  `identity array`

# In[60]:


arr_identity = np.identity(10, dtype='int8')


# In[61]:


print(arr_identity)


# ## Indexing and Slicing in NumPy Array

# ### 1-D indexing & slicing

# In[62]:


arr_1d


# #### Indexing

# In[63]:


arr_1d[0] #0th position


# In[66]:


arr_1d[-1] #last position using reverse index


# In[67]:


arr_1d[-5] #first position but using reverse index


# #### slicing

# `start : stop : step`

# In[68]:


arr_1d[:] #shows the full array


# In[69]:


arr_1d[::] #shows the full array


# In[70]:


arr_1d[: : 2] #alternate item


# ### 2-D indexing & slicing

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# In[73]:


arr_2d = np.array([[1,2,3], [4,5,6], [7,8,9]])


# In[74]:


arr_2d


# In[75]:


arr_2d.ndim


# ![image.png](attachment:image.png)

# `i` selects the `row` and `j` selects the `column`
# 
# `:` colon means every element in row/column

# In[76]:


arr_2d[:]


# In[77]:


arr_2d[:,:]


# In[78]:


arr_2d[: :,: :]


# In[79]:


arr_2d[0] # first row


# In[80]:


arr_2d[0,0] # first element


# In[81]:


arr_2d[1,1] # midddle element


# In[83]:


arr_2d[2, 0]


# In[85]:


arr_2d[:,:] #here colon means everything from rows & columns


# In[86]:


arr_2d[0,1]  #first row and 2nd column


# #### second column

# In[90]:


arr_2d[:, 1]


# In[88]:


arr_2d[:, -1] #last column


# ![image.png](attachment:image.png)

# In[94]:


arr_2d[:,:2] #first two columns


# In[95]:


arr_2d[:,: : 2] #alternating columns


# #### reversing 2D array

# `by row`

# In[97]:


arr_2d[: : -1, :]


# `by column`

# In[98]:


arr_2d[:, : : -1]


# ####  Transposing the array

# In[99]:


arr_2d


# In[100]:


arr_2d.T


# #### Show the principal diagonal of the 2D array

# `Hint: np.diag()`
# 
# `Expected output: [1 5 9]`

# In[102]:


np.diag(arr_2d)


# ### Indexing and slicing in 3 dimensions

# ![image.png](attachment:image.png)

# In[104]:


arr_3d = np.array([
    [[10,11,12],[13,14,15], [16,17,18]],    #first layer-2D
    [[20,21,22],[23,24,25], [26,27,28]],    #second layer-2D
    [[30,31,32],[33,34,35], [36,37,38]]    #third layer-2D
])


# In[105]:


arr_3d


# #### Print the first layer

# ![image.png](attachment:image.png)

# In[107]:


arr_3d[0] #by default means the first layer


# In[108]:


arr_3d[-1] #by default means the last layer


# In[109]:


arr_3d[1, : , :] #middle layer by slicing


# In[110]:


arr_3d[1] #middle layer by indexing


# #### Q. Print the middle column of the middle layer

# ![image.png](attachment:image.png)

# In[111]:


arr_3d[1,:,1]


# #### Q. Print the middle column across the layers

# In[112]:


arr_3d[:,:,1]


# ### H/W. Print the prinicpal diagnoal elements across the array

# ![image.png](attachment:image.png)

# `10,14,18 | 20,24,28 | 30,34,38`

# https://www.kaggle.com/code/themlphdstudent/learn-numpy-numpy-50-exercises-and-solution

# ### Array Mathematics

# ![image.png](attachment:image.png)

# #### addition
# `within the array`

# In[113]:


arr_1d


# In[115]:


np.sum(arr_1d) #sum of all the elements


# `tab` : `shows all the possible funtions with that keyword`
#     
# `shift + tab`: `shows the documentation / syntax for the selected function`
# 

# In[116]:


arr_2d


# In[117]:


np.sum(arr_2d) #sum of all the elements


# In[118]:


np.sum(arr_2d,axis=0) #sum of all the elements along the rows


# In[119]:


np.sum(arr_2d,axis=1) #sum of all the elements along the columns


# #### addition
# `two or more arrays`

# ### 1D

# In[122]:


arr_1 = np.array([1,2,3])
print(arr_1)
arr_2 = np.array([9,8,7])
print(arr_2)


# #### Elementwise addition

# In[123]:


arr_1 + arr_2


# ### 2D

# In[124]:


arr_2d


# In[125]:


arr_2d2 = np.array([[99,98,97],
                   [96,95,94],
                   [93,92,91]])

print(arr_2d2)


# #### summing arr_2d with arr_2d2 - elementwise

# In[127]:


arr_2d + arr_2d2


# #### summing arr_2d with arr_2d2 - elementwise after doing sun along the rows

# In[128]:


np.sum(arr_2d, axis=0)


# In[129]:


np.sum(arr_2d2, axis=0)


# In[130]:


np.sum(arr_2d, axis=0) + np.sum(arr_2d2, axis=0)


# In[132]:


np.sum((arr_2d + arr_2d2), axis=0)


# In[133]:


np.sum((arr_2d[:,1] + arr_2d2[:,1]), axis=0)


# In[135]:


arr_2d[:,1] + arr_2d2[:,1]


# In[136]:


arr_1 + arr_2d


# In[137]:


arr_1


# In[138]:


arr_2d


# In[139]:


arr_4 = np.array([1,2,3,4])


# In[140]:


arr_4 + arr_2d


# ### UFunctions - > Universal Functions
# `pro tip: interview`

# - Universal function works on a single input
# - Binary UFs on two inputs

# ![image.png](attachment:image.png)

# In[141]:


arr_1d


# In[142]:


print('Add 5 to arr_1d:', arr_1d + 5)


# In[143]:


np.sum(arr_1d + 5)


# `np.sum()` : computes the sum of array elements along a specified axis (2D or more) / sums all the elements of the array
#     
# `np.add()`: performs element-wise addition between two arrays

# In[144]:


my_list= [1,10,12,16,20]


# In[145]:


my_list + 5


# In[146]:


my_list + [5] #concatenation


# ### H/W Reading assignment: BROADCASTING in Arrays

# In[150]:


x =np.array([2,4,6])


# In[151]:


x


# In[152]:


print('x =', x)
print('x+5 = ', x+5)  #adding 5
print('x-5 = ', x-5) #subtracting 5
print('x*2 = ',x*2) #multiply 2
print('x/2 = ', x/2)#divide by 2
print('x//2 = ', x//2)#floor division by 2
print('-x =', -x) #negation
print('x**2 =', x**2) #power 2
print('x%2 = ', x%2) #remainder


# In[153]:


print('x+5 = ', x+5)  #adding 5


# In[155]:


np.add(x, 5)


# In[156]:


np.subtract(x, 5)


# ### H/W: Demonstrate all universal functions shown above as part of arithmetic operations

# ### Math Functions

# In[162]:


arr_math = np.array([2,4,16])


# In[163]:


print(arr_math)


# In[165]:


print('Square root of the array:', np.sqrt(arr_math))


# In[166]:


print('Natural log of the array:', np.log(arr_math)) #base e


# In[167]:


print('Log of the array:', np.log10(arr_math)) #base 10


# In[169]:


print('Sine of the array:', np.sin(arr_math)) #sin theta


# In[170]:


print('Cosine of the array:', np.cos(arr_math)) #Cosine theta


# ### Basic Statistics

# In[172]:


python_scores = np.array([10,20,15,20,12,15,8,5,13,17])
print(python_scores)


# In[183]:


print('Sum:', python_scores.sum()) #sum
print('Mean / Avg. Python Scores:', python_scores.mean())
print('Max Python Scores:', python_scores.max())
print('Min Python Scores:', python_scores.min())
#print('Median Python Scores:', python_scores.median()) # this function is not there
print('Median Python Scores:', np.median(python_scores))
print('Variance Python Scores:', np.var(python_scores))
print('Variance Python Scores:', python_scores.var())
print('Standard deviation Python Scores:', python_scores.std())


# **Mode is not directly available with Numpy**

# ### H/W Try finding mode using Pandas

# In[184]:


from scipy import stats


# In[186]:


mode_result = stats.mode(python_scores)


# In[187]:


mode_result.mode


# In[188]:


mode_result.count


# ![image.png](attachment:image.png)

# -- DO NO USE XXXX

# ### Let us write a function to calculate mode 

# In[189]:


np.unique(python_scores, return_counts=True)


# In[213]:


def find_modes(arr):
    unique_value, value_count = np.unique(arr, return_counts=True)
    max_count = np.max(value_count) #maximum count
    if max_count>1:
        modes = unique_value[value_count==max_count]
        return modes, max_count
    else:
        print("No mode found")
        return None, None


# In[214]:


modes, count = find_modes(python_scores)
print("Mode value(s) are:", modes)
print("Frequency of the Mode Value(s) are:", count)


# In[215]:


python_scores_2 = np.array([10,20,15,20,12,15,8,5,13,17, 20, 20, 10,10])
print(python_scores_2)


# In[216]:


modes, count = find_modes(python_scores_2)
print("Mode value(s) are:", modes)
print("Frequency of the Mode Value(s) are:", count)


# In[217]:


python_scores_3 = np.array([10,11,12,13,5,9,20,18,19])
print(python_scores_3)


# In[218]:


modes, count = find_modes(python_scores_3)
print("Mode value(s) are:", modes)
print("Frequency of the Mode Value(s) are:", count)


# ## Create an array: np.linspace & np.arange

#  `pro tip: interview question`

# ![image.png](attachment:image.png)

# - When it comes to create a sequence of values, `linspace` and `arange` are two commonly used NumPy functions

# Here is the subtle difference between the two functions:
# 
# - `linspace` allows you to specify the **number of values**
# - `arange` allows you to specify the **size of the step**

# #### np.linsapce()

# `np.linspace(start, stop, num, …)`
# 
# where:
# 
# - start: The starting value of the sequence
# - stop: The end value of the sequence
# - **num: the number of values to generate**

# In[1]:


import numpy as np


# In[19]:


np.linspace(1,100, 13,retstep=True, dtype='int32') #start and stop are included


# - By default, 50 numbers will be generated
# - `np.linspace` returns 1-D array and to get into other dimensions we need to use array `reshaping`

# **Using this method, np.linspace() automatically determines how far apart to evenly space the values**

# #### np.arange()

# `np.arange(start, stop, step, …)`
# 
# where:
# 
# - start: The starting value of the sequence
# - stop: The end value of the sequence
# - **step: the spacing between the values**

# In[23]:


np.arange(10,50,10) #it doesn't include the stop (50)


# In[24]:


np.arange(10,50.00000001,10) #it doesn't include the stop (50.0000001)


# #### Print all the even numbers between `0` and `101`

# In[27]:


np.arange(0,101,2)


# ## Array Manipulation

# `  **resize()** and **reshape()**

# `pro tip: interview question`

# #### resize()

# - returns a **new** array with the specified shape
# - if the new array is larger than the original array, the new array is going to be filled with the repeated copies of the original array

# ![image.png](attachment:image.png)

# In[29]:


a = np.array([[1,2],[3,4]])
print(a) #base / original array


# In[30]:


np.resize(a, (3,2))


# In[31]:


np.resize(a, (5,2))


# In[32]:


np.resize(a, (3,3,3))


# In[34]:


# np.resize(a, (10,10,10))


# In[39]:


b = np.array([1,2,3,5,7,8,11,13,16,18,20])


# In[40]:


print(b)


# In[41]:


new_values = [7,9,100]
inserted_arr = np.insert(b,[3,6,8], new_values )
print(inserted_arr)


# #### reshape()

# - is used to a give a new shape to an array `without changing its data/elements`
# - the new shape must be compatible with the original shape

# In[42]:


a


# In[44]:


print(np.reshape(a, (4,1)))


# In[45]:


print(np.reshape(a, (1,4)))


# In[48]:


arr_3d = np.array([
    [[10,11,12], [13,14,15], [16,17,18]], #first layer - 2D
    [[20,21,22], [23,24,25], [26,27,28]], #second layer - 2D
    [[30,31,32], [33,34,35], [36,37,38]] #third layer - 2D
])

print(arr_3d)


# In[49]:


print(np.reshape(arr_3d, (1,9,3)))


# In[50]:


print(np.reshape(arr_3d, (1,3,9)))


# In[51]:


print(np.reshape(arr_3d, (9,3))) #2d array


# ## STACKING

# - Number of rows/columns does not need to be same for stacking

# ![image.png](attachment:image.png)

# ### 1-D Stacking

# In[52]:


a = np.array([1,2,3])
print(a)


# In[53]:


b = np.array([12,14,16])
print(b)


# #### Stacking one on the top of the other (axis 0)

# In[54]:


ab_stacked_axis_0  = np.stack((a,b), axis=0)


# In[55]:


ab_stacked_axis_0


# In[56]:


ab_stacked_axis_0.ndim


# #### Stacking side by side(axis 1)

# In[58]:


ab_stacked_axis_1 = np.stack((a,b), axis=1)


# In[59]:


ab_stacked_axis_1


# In[60]:


ab_stacked_axis_1.ndim


# ### 2-D stacking

# In[61]:


a_2d = np.array([[1,2,3], [4,5,6]])
print(a_2d)
print(a_2d.ndim)


# In[62]:


b_2d = np.array([[10,20,30], [40,50,60]])
print(b_2d)
print(b_2d.ndim)


# In[66]:


#stacking one on the top of the other
ab_stacked_axis_0=np.stack((a_2d, b_2d), axis=0)


# In[67]:


ab_stacked_axis_0


# In[65]:


ab_stacked_axis_0.ndim


# In[68]:


#stacking side by side
ab_stacked_axis_1=np.stack((a_2d, b_2d), axis=1)
ab_stacked_axis_1


# ### Concatenation

# In[69]:


a #1D


# In[70]:


b


# In[71]:


np.concatenate((a,b), axis=0) #1D


# In[72]:


np.concatenate((a,b), axis=1) 


# In[73]:


np.concatenate((a_2d, b_2d),axis=0) #2D


# ### hstack vs vstack

# #### hstack

# ![image.png](attachment:image.png)

# - NumPy stack function takes 2 arrays with the same number of rows and joins them automatically

# In[74]:


a1 = np.array([[1,2],
              [3,4]])

print(a1)
print(a1.shape)


# In[75]:


b1 = np.array([[10,20,30],
              [40,50,60]])

print(b1)
print(b1.shape)


# - Observation: Number of rows is same

# In[76]:


np.hstack([a1,b1]) #2D


# In[78]:


c1 = np.array([[10,20,30],
              [40,50,60],
              [99,100,101]])

print(c1)
print(c1.shape)


# In[79]:


np.hstack([a1,c1])


# `In a1 and c1, numbers of rows not matching`

# #### vstack

# - NumPy vstack funtion takes 2 arrays with the `same number of columns` and joins them vertically

# ![image.png](attachment:image.png)

# In[83]:


d1=np.array([[99,98,97],
         [80,90,100]])
print(d1)
print(d1.shape)


# In[84]:


c1.shape


# In[85]:


np.vstack([c1,d1])


# In[87]:


np.vstack([c1,a1])


# ### Splitting the arrays

# #### np.vsplit()

# ![image.png](attachment:image.png)

# In[88]:


arr1 = np.array([[1,2,2],
                [2,0,0],
                [3,1,1],
                [4,0,4]])

print(arr1)
print(arr1.shape)


# In[90]:


np.vsplit(arr1,2) #splits it into two


# In[91]:


p1, p2 = np.vsplit(arr1,2) #splits it into two


# In[92]:


p1


# In[93]:


p2


# In[95]:


np.vsplit(arr1,4) #splits it into four


# In[97]:


np.vsplit(arr1,3) #splits it into three


# #### np.hsplit()

# In[99]:


arr2 = np.linspace(10,100,16)


# In[100]:


arr2


# In[101]:


arr2 = np.reshape(arr2, (4,4))
arr2


# In[102]:


np.hsplit(arr2, 2)


# In[103]:


np.hsplit(arr2, 4)


# In[104]:


np.hsplit(arr2, 3)


# ### Broadcasting

# - Broadcasting is an approach in NumPy that allows arrays with different shapes to be used in arithmetic operations
# - It eliminates the need for replicating arrays to match shapes before performing operations
# - Broadcasting has some set of rules to align the shape of the arrays automatically

# ![image.png](attachment:image.png)

# In[105]:


arr2


# In[106]:


arr3 = np.array([1,2,3,4])


# In[107]:


arr3


# In[108]:


arr2.shape


# In[109]:


arr3.shape


# In[110]:


arr2+arr3


# In[112]:


arr4 = arr_3d[1,:,:]


# In[113]:


arr4


# In[114]:


arr4.shape


# In[115]:


arr4+arr3


# ![image.png](attachment:image.png)

# - **When the trailing dimensions of the arrays are unequal, broadcasting fails**

# In[ ]:




