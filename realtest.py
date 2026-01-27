import tkinter as tk

root = tk.Tk()

# Create a canvas widget
# Width, height, and background color are common options
canvas = tk.Canvas(root, width=410, height=310, bg="white")
canvas.pack()

# Draw a rectangle: provide top-left (x1, y1) and bottom-right (x2, y2) coordinates
h = 310
w = 410
n_perlayer = [1,2,3,2,9,3,1]
layer = 7

b = 0.03

a = 0.5

size_y = h/max(n_perlayer)
size_x = w/(layer*(1+a*2))

size = min([size_x,size_y])
    
print([size_x,size_y])

for j in range(layer) :
    for i in range(n_perlayer[j]) :
        space = (h-(n_perlayer[j]*size*(1+b)))/(n_perlayer[j]+1)
        
        canvas.create_oval(j*size+(size*b)+(size*a*(j*2+1)),space + i*(space+size*(1+b)),(j+1)*size-(size*b)+(size*a*(j*2+1)), (i+1)*(space+size*(1+b)))
        
        c = ((j+1)*size+(size*b)+(size*a*(j*2+1)),space + i*(space+size*(1+b)) + size*(1+b)/2)
            
        if j != layer-1 :
            space_next = (h-(n_perlayer[j+1]*size*(1+b)))/(n_perlayer[j+1]+1)
            for k in range(n_perlayer[j+1]) :
                canvas.create_line(c[0],c[1],(j+1)*size+(size*b)+(size*a*((j+1)*2+1)), space_next + (size/2) + k*(space_next+size*(1+b)),fill='blue')
                

# Draw a line: provide start and end coordinates
# canvas.create_line(0, 200, 1000, 200, fill="green", width=2)
# canvas.create_line(310, 0, 310, 1000, fill="green", width=2)
# canvas.create_line(0, 100, 1000, 100, fill="green", width=2)
# canvas.create_line(100, 0, 100, 1000, fill="green", width=2)


# Run the Tkinter event loop
root.mainloop()
