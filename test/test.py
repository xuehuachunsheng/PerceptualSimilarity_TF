

a = 'AkjdasASKDJHkasdhAjsdDjkDkajksdhqKASld'
a = 'aAkjdasASKDJHka'
a = list(a)
min = 0
max = 0
i = 0

for i in range(len(a)):
    if a[i].islower():
        c_lower = a[i]
        j = i - 1
        while j >= 0 and a[j].isupper():
            a[j + 1] = a[j]
            j -= 1
        if j >= -1:
            a[j + 1] = c_lower

print(''.join(a))



