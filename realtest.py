import tkinter as tk

root = tk.Tk()

# Create a canvas widget
# Width, height, and background color are common options
canvas = tk.Canvas(root, width=410, height=310, bg="white")
canvas.pack()

# Draw a rectangle: provide top-left (x1, y1) and bottom-right (x2, y2) coordinates
h = 310
w = 410
n_perlayer = 10
layer = 7

b = 0.03

a = 0.5

size_y = h/n_perlayer
size_x = w/(layer*(1+a*2))

if size_x < size_y :
    size = size_x
else :
    size = size_y

for j in range(layer) :
    for i in range(n_perlayer) :
        canvas.create_oval(j*size+(size*b)+(size*a*(j*2+1)), i * size+(size*b),(j+1)*size-(size*b)+(size*a*(j*2+1)), (i+1) * size-(size*b))
        
        
        # print(j*size, i * size,(j+1)*size, (i+1) * size)
        
        if j != layer-1 :
            for k in range(n_perlayer) :
                canvas.create_line((j+1)*size-(size*b)+(size*a*(j*2+1)), (i+0.5) * size,(j+2)*size+(size*b)+(size*a*(j*2+1)), (k+0.5) * size, fill="green", width=2)
                print((j+1)*size-(size*b)+(size*a*(j*2+1)), (i+0.5) * size, (j+1)*size+(size*b)+(size*a*(j*2+1)), (k+0.5) * size)

# Draw a line: provide start and end coordinates
# canvas.create_line(0, 200, 1000, 200, fill="green", width=2)
# canvas.create_line(310, 0, 310, 1000, fill="green", width=2)
# canvas.create_line(0, 100, 1000, 100, fill="green", width=2)
# canvas.create_line(100, 0, 100, 1000, fill="green", width=2)


# Run the Tkinter event loop
root.mainloop()
