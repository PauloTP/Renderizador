# Desenvolvido por: Luciano Soares <lpsoares@insper.edu.br>
# Disciplina: Computação Gráfica
# Data: 28 de Agosto de 2020

import argparse     # Para tratar os parâmetros da linha de comando
import x3d          # Faz a leitura do arquivo X3D, gera o grafo de cena e faz traversal
import interface    # Janela de visualização baseada no Matplotlib
import gpu          # Simula os recursos de uma GPU


#newimports
import math
import numpy as np

#parte 1 comeca agora:
def polypoint2D(point, color):
    """ Função usada para renderizar Polypoint2D. """
    #gpu.GPU.set_pixel(3, 1, 255, 0, 0) # altera um pixel da imagem
    #point is a list (x1,y1,x2,y2,x3,y2)
    #color = (r,g,b. de 0 a 1) (tem q converter a 255)
    # cuidado com as cores, o X3D especifica de (0,1) e o Framebuffer de (0,255)

    bitcolor = []
    for i in color:
        bitcolor.append(int(i * 255))

    i = 0
    while i < (len(point)):
        x_round = math.floor(point[i])
        y_round = math.floor(point[i+1])
        if x_round >= width:
            x_round = width-1
        if y_round >= height:
            y_round = height-1
        gpu.GPU.set_pixel(x_round, y_round, bitcolor[0], bitcolor[1], bitcolor[2]) # altera um pixel da imagem
        i+=2

def polyline2D(lineSegments, color):
    """ Função usada para renderizar Polyline2D. """
    #x = gpu.GPU.width//2
    #y = gpu.GPU.height//2
    #gpu.GPU.set_pixel(x, y, 255, 0, 0) # altera um pixel da imagem
    print(lineSegments)
    
    bitcolor = []
    for i in color:
        bitcolor.append(int(i * 255))
    
    #x = u
    #y = v
    #getting values for readbility
    if(lineSegments[0] < lineSegments[2]):
        v1 = lineSegments[1]
        v2 = lineSegments[3]
        u1 = lineSegments[0]
        u2 = lineSegments[2]
    else:
        v2 = lineSegments[1]
        v1 = lineSegments[3]
        u2 = lineSegments[0]
        u1 = lineSegments[2]

    #calc angular coeficient
    if(u2 - u1 == 0):
        s = math.inf
    else:
        s = (v2-v1) / (u2-u1)
    #print(s)
    #calc angular coef and set u and v
    u = u1
    v = v1

    if(math.fabs(s) <= 1):
        while(u <= u2):
            gpu.GPU.set_pixel(math.floor(u), math.floor(v), bitcolor[0], bitcolor[1], bitcolor[2])
            v += s
            u += 1

    else: #if line is vertical, change the logic a bit
        s = 1/s
        #required in some cases
        if(v2 < v1):
            u = u2
            v = v2
            while(v <= v1):
                gpu.GPU.set_pixel(math.floor(u), math.floor(v), bitcolor[0], bitcolor[1], bitcolor[2])
                v += 1
                u += s
        else:
            while(v <= v2):
                gpu.GPU.set_pixel(math.floor(u), math.floor(v), bitcolor[0], bitcolor[1], bitcolor[2])
                v += 1
                u += s

#funcs for triangle:
def dot(a,b):
    #a and b are 2sized float lists
    return a[0]*b[0] + a[1]*b[1]

def get_angle(vec1, vec2): 
    #both vectors have a point in common (the second point)
    a = [vec1[0],vec1[1]]
    b = [vec1[2],vec1[3]]
    c = [vec2[0],vec2[1]]

    #line adapted from: https://medium.com/@manivannan_data/find-the-angle-between-three-points-from-2d-using-python-348c513e2cd
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    ret = ang + 360 if ang < 0 else ang

    if ret > 180:
        ret = 360 - ret
    return ret

def check_point(vertices, point):
    #if the sum of the angle of the vectors between the point and each vertices is 360, the point is inside
    vectors = []
    '''
    for i in range(0,len(vertices),2):
        vectors.append(vertices[i] - point[0])
        vectors.append(vertices[i+1] - point[1])
    '''
    for i in range(0,len(vertices),2):
        v = [vertices[i],vertices[i+1],point[0],point[1]]
        vectors.append(v)
    #print(vectors)
    #DEBUG
    #polyline2D(vectors[0],[1,1,1])
    #print(get_angle(vectors[0],vectors[1]))
    angles = []
    angles.append(get_angle(vectors[0],vectors[1]))
    angles.append(get_angle(vectors[1],vectors[2]))
    angles.append(get_angle(vectors[2],vectors[0]))
    
    #config

    THRESHOLD = 0.02 #deegres. This set the amount of error that still confirms a point is inside a triangle

    anglesum = sum(angles)

    return (anglesum < (360 + THRESHOLD) and anglesum > (360 - THRESHOLD))

def draw_pixel(vertices, color, pixel):

    if (pixel[0] >= LARGURA) or (pixel[1] >= ALTURA) or (pixel[0] < 0) or (pixel[1] < 0):
        return "OUT OF BOUNDS" #flag for not trying to render anything outside the camera

    #DEBUG: White background (sets the pixel to white before checking it, to see witch pixels are being rendered and debuging)
    #gpu.GPU.set_pixel(pixel[0], pixel[1], 255, 255, 255)

    
    intensity = 0 #this is returned to know how much of the samples were in the triangle

    #Eh um antialising! Eh uma otimizacao em GPU! NAO!!! Eh o SUPER SAMPLING!!!
    SSP = [
            [pixel[0] + 0.3, pixel[1] + 0.3],
            [pixel[0] + 0.7, pixel[1] + 0.3],
            [pixel[0] + 0.3, pixel[1] + 0.7],
            [pixel[0] + 0.7, pixel[1] + 0.7]
        ]

    for i in SSP:
        if check_point(vertices,i):
            intensity += 1 / len(SSP)
    
    bitcolor = []
    for i in color:
        bitcolor.append(int((i) * 255))

    if(intensity == 0):
        return False
    
    #if triangle is above another rendered object
    old_color = gpu.GPU.get_pixel(pixel[0],pixel[1])

    #OLD Linear transparency calc.
    #r = old_color[0] * (1 - intensity) + bitcolor[0] * intensity
    #g = old_color[1] * (1 - intensity) + bitcolor[1] * intensity
    #b = old_color[2] * (1 - intensity) + bitcolor[2] * intensity

    #quadratic normalized transparency
    r = ( (old_color[0]**2) * (1 - intensity) + (bitcolor[0]**2) * intensity )**0.5
    g = ( (old_color[1]**2) * (1 - intensity) + (bitcolor[1]**2) * intensity )**0.5
    b = ( (old_color[2]**2) * (1 - intensity) + (bitcolor[2]**2) * intensity )**0.5

    #unfortunally the class GPU does not support an alpha channel Dx

    gpu.GPU.set_pixel(pixel[0], pixel[1], r, g, b)

    return True


def triangleSet2D(vertices, color):
    """ Função usada para renderizar TriangleSet2D. """
    #gpu.GPU.set_pixel(24, 8, 255, 255, 0) # altera um pixel da imagem
    print(vertices)
    #print(check_point(vertices,[10,10]))

    #check every pixel
    '''
    for i in range(30):
        for j in range(20):
            draw_pixel(vertices,color,[i,j])
    '''

    #pseudo zig-zag
    #start on highest vertice
    #go right until end of triangle
    #go left until end of triangle
    #go down one
    #search triangle
    #repeat until below lowest Y

    #get highest vertice (high is a lower Y value)
    highest = ALTURA
    lowest = 0
    high_id = 0

    print("debug:",vertices)

    for i in range(3):
        y = vertices[(i*2) + 1]
        print("debug_y:",y)
        if(y < highest):
            highest = y
            high_id = i
        if (y > lowest):
            lowest = y
    #start on it:

    pixel = [math.floor(vertices[high_id*2]), math.floor(vertices[high_id*2 + 1])]
    lowest = math.floor(lowest)
    print("pixel mais alto: " ,pixel)
    print("y mais baixo: " , lowest)

    #start loop:
    origin = pixel[0]
    line = pixel[1]
    delta = 0
    lost = False
    full = False
    state = "down" #right, left, down, search

    render_count = 0 
    #DEBUG
    speed = 10
    Out_of_bounds = False
    while line < lowest and (not Out_of_bounds):
        #THERE IS NO SWITCH CASE IN PYTHON!!! (sad music starts playing)
        drew = draw_pixel(vertices,color,[origin + delta, line])
        if(drew == "OUT OF BOUNDS"):
            Out_of_bounds = True
            render_count += 1
            print("OUT OF BOUNDS")
            #I went out of bounds with my triangle. so lets finish the screen with a brute force method for now:
            if(line < 0):
                line = 0
            while line <= lowest and line <= ALTURA:
                for i in range(LARGURA):
                    render_count += 1
                    draw_pixel(vertices,color,[i, line])
                line += 1
            #End of nojeira de ultimo caso

            continue
        render_count += 1
        #If you want to see this algo running, uncomment this speed if and the first DEBUG line on Draw_pixel (white bg)
        '''
        if speed == 0:
            interface.Interface(LARGURA, ALTURA, image_file).preview(gpu.GPU._frame_buffer)
            speed = 10
        else:
            speed -= 1
        '''
        if state == "down":
            if drew:
                state = "right"
                delta = 1
            else:
                state = "search"
                origin += 1
            #end state down
        elif state == "right":
            if drew:
                delta += 1
            else:
                state = "left"
                delta = -1
            #end state right
        elif state == "left":
            if drew:
                origin -= 1
                lost = False
            else:
                if lost:
                    state = "search"
                    line += 1
                else:
                    state = "down"
                    delta = 0
                    line += 1
            #end state left
        elif state == "search":
            if drew:
                state = "right"
                delta = 1
                lost = False
                full = False
            else:
                if(line > lowest):
                    break
                else:
                    if(lost):
                        if full:
                            #this is a last case scenario for weird triangles
                            origin += 1
                            if (origin > LARGURA):
                                origin = 0
                                line+= 1

                        elif(origin == 0):
                            #scenario: Long thin Triangle... also F this triangle
                            #this case, search every pixel until find one that is drawable
                            line += 1
                            full = True

                        else:  
                            origin -= 1
                    else:
                        lost = True
                        origin -= 1
                        delta = -1
                        state = "left"
            #end state search

        #end of state machine

    print ("Done")
    print (f"Rendered {render_count} pixels")


    #acima disso eh a parte 1

#vou criar uma classe para armazenar as operacoes de transform e o que mais for nescessario e ela ser percorrida como uma pequena
#pipeline para cada ponto a ser renderizado. porem o pipeline em si muda com o passar do codigo.
#-so estou preocupado com como o python vai lidar com o passar de funcoes em um dic para outra classe... espero que eu ainda tenha acesso
#ao escopo dessa classe de dentro da x3d.py :T
class Procedure():
    def __init__(self):
        self.operation_queue = []
        self.queue_size = 0
        self.fov = 0
    
    def add_operation(self, op_type, metadata):
        #op_type is a type of operation like Rotation, Scale, Translation, etc.
        #metadata is a generic var that is different for every operation. (usually a simple list)
        if self.queue_size >= len(self.operation_queue):
            self.operation_queue.append([op_type, metadata])
            self.queue_size = len(self.operation_queue) 
        else:
            self.operation_queue[self.queue_size] = [op_type, metadata]
            self.queue_size += 1
        #the choice of using a queue_size var is to easily set the last valid operation, thus making removing the last operation easy

    def pop_operation(self):
        self.queue_size -= 1

    #Field of View function is not an operation, as the user should not be allowed to apply it before any other operation
    #or apply it more than once.
    #It could be added to the operation queue but Having it as a separated thing seems more sane.
    def apply_fov(self, point):
        #print(f"Applying FOV for {point} with fov of {procedure.fov}")

        near = 1
        far = 100
        top = near * math.tan(procedure.fov)
        bot = -top
        right = top * (LARGURA/ALTURA)
        left = -right

        #matrix form of point (a "tall" matrix)
        point_matrix = np.matrix([  [point[0]],
                                    [point[1]],
                                    [point[2]],
                                    [1]          ])

        #translation matrix
        fov_matrix = np.matrix([ [near/right, 0, 0, 0],
                                   [0, near/top, 0, 0],
                                   [0, 0, -((far + near) / (far - near)), (-2 * far * near) / (far - near)],
                                   [0, 0, -1, 0]  ])
        #multiply!!!
        result = np.matmul(fov_matrix, point_matrix)
        result = result.tolist()
        point = [ result[0][0], result[1][0], result[2][0] ]

        #print(point)

        return point

    #defining every possible operation as separated functions:""
    def solve_translation(point, metadata):
        #Metadata is a vector [x,y,z] and this function moves the point by this vector
        #print(f"Solving translation for {point} with {metadata}")
        
        #matrix form of point (a "tall" matrix)
        point_matrix = np.matrix([  [point[0]],
                                    [point[1]],
                                    [point[2]],
                                    [1]          ])

        #translation matrix
        Mx, My, Mz = metadata
        trans_matrix = np.matrix([ [1, 0, 0, Mx],
                                   [0, 1, 0, My],
                                   [0, 0, 1, Mz],
                                   [0, 0, 0, 1]  ])
        #multiply!!!
        result = np.matmul(trans_matrix, point_matrix)
        result = result.tolist()
        point = [ result[0][0], result[1][0], result[2][0] ]

        return point

    def solve_scale(point, metadata):
        #print(f"Solving scale for {point} with {metadata}")

        #matrix form of point (a "tall" matrix)
        point_matrix = np.matrix([  [point[0]],
                                    [point[1]],
                                    [point[2]],
                                    [1]          ])

        #scale matrix
        Sx, Sy, Sz = metadata
        scale_matrix = np.matrix([ [Sx, 0, 0, 0],
                                   [0, Sy, 0, 0],
                                   [0, 0, Sz, 0],
                                   [0, 0, 0, 1]  ])
        #multiply!!!
        result = np.matmul(scale_matrix, point_matrix)
        result = result.tolist()
        point = [ result[0][0], result[1][0], result[2][0] ]

        return point

    def solve_rotation(point, metadata):
        #print(f"Solving rotation for {point} with {metadata}")
        #matrix form of point (a "tall" matrix)
        point_matrix = np.matrix([  [point[0]],
                                    [point[1]],
                                    [point[2]]   ])

        #rotation matrix
        cos,sin = [math.cos(metadata[3]), math.sin(metadata[3])]

        if metadata[0] >= 1:
            #Rotate on X
            rot_matrix = np.matrix([ [1, 0, 0],
                                     [0, cos, -sin],
                                     [0, sin, cos] ])
        elif metadata[1] >= 1:
            #Rotate on Y
            rot_matrix = np.matrix([ [cos, 0, sin],
                                     [0, 1, 0],
                                     [-sin, 0, cos] ])
        elif metadata[2] >=1:
            #rotate on Z
            rot_matrix = np.matrix([ [cos, -sin, 0],
                                     [sin, cos, 0],
                                     [0, 0, 1] ])
        
        #multiply!!!
        result = np.matmul(rot_matrix, point_matrix)
        result = result.tolist()
        point = [ result[0][0], result[1][0], result[2][0] ]
        #print(point)
        return point

    #put all operation on a single dictionary
    operation = {}
    operation["Translation"] = solve_translation
    operation["Rotation"] = solve_rotation
    operation["Scale"] = solve_scale

    def run_procedure(self, point):

        #
        #a Sessao comentada abaixo veio de uma versao anterior, deixei aqui para analise do professor:
        #for op in self.operation_queue:
            #chamando a operacao baseada no tipo (dado por op[0]) e passando o ponto e os dados de metada (op[1])
            #cada iteracao atualiza o valor de point, como se fosse uma chamada recursiva
            #nao fiz recursivamente por que dessa forma eh mais legivel, caso contrario precisaria iterar de traz pra frente e fazer as
            #chamadas em uma stack q seria resolvida da primeira operacao da fila ate a ultima e seria estranho...
            #point = self.operation[op[0]](point, op[1])
        
        #Versao correta (Pilha) 
        for i in range(self.queue_size -1, -1, -1):
            #chamando a operacao baseada no tipo (dado por op[0]) e passando o ponto e os dados de metada (op[1])
            #cada iteracao atualiza o valor de point, como se fosse uma chamada recursiva
            op = self.operation_queue[i]
            point = self.operation[op[0]](point, op[1])

        #Debug and experimentation
        #I left these lines here to play a little with rotation and scale
        #and ultimately left the scale at 2.5, for visibility
        #also, if you scale it by like, 10, and increase the resolution, you can have a more clear picture
        #(specially usefull in the box on example 6)
        point = self.operation["Rotation"](point, [0,0,1,0])
        point = self.operation["Scale"](point, [2.5,2.5,2.5])


        #Applying FOV to each point AFTER all other operations
        point = self.apply_fov(point)
        
        #screen transformations:
        #Flip on Y. So the projection is not mirrored
        point = self.operation["Scale"](point, [1,-1,1])
        #making a last translation that happens after everything to simply set to origin UV on the center of the camera
        point = self.operation["Translation"](point, [LARGURA / 2, ALTURA / 2, 0])

        return point

#create procerdure object (python abstract classes are wierd, so it is better to make an instance)
procedure = Procedure()


def triangleSet(point, color):
    """ Função usada para renderizar TriangleSet. """
    # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
    # de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x do
    # primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da 
    # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e
    # assim por diante.
    # No TriangleSet os triângulos são informados individualmente, assim os três
    # primeiros pontos definem um triângulo, os três próximos pontos definem um novo
    # triângulo, e assim por diante.

    #applying the procedure to convert the points to the camera local space, and also applying FOV
    point_list = []
    for i in range(0, len(point), 3):
        point_list.append(procedure.run_procedure( [point[i], point[i+1], point[i+2]] ) )

    #print("TriangleSet : pontos = {0}".format(point_list)) # imprime no terminal pontos

    #converting to uv:
    #(basically ignoring Z as everything is in the camera local space after the procedures)
    vertices = []
    for i in range(0, len(point_list)):
        vertices.append(point_list[i][0])
        vertices.append(point_list[i][1])

    #print("TriangleSet : uv = {0}".format(vertices)) # imprime no terminal pontos
    for i in range(0, len(vertices), 6):
        #render each triangle
        triangleSet2D(vertices[i:i+6],color)


def viewpoint(position, orientation, fieldOfView):
    """ Função usada para renderizar (na verdade coletar os dados) de Viewpoint. """
    # Na função de viewpoint você receberá a posição, orientação e campo de visão da
    # câmera virtual. Use esses dados para poder calcular e criar a matriz de projeção
    # perspectiva para poder aplicar nos pontos dos objetos geométricos.

    #save FOV for later
    procedure.fov = fieldOfView
    #add translation operation for bringing everything to the camera local space
    procedure.add_operation("Translation", [-position[0], -position[1], -position[2]] )
    #the viewpoint orientation can be ignored for this project, as we are working with euler angles instead of quaternions
    #so we can only rotate one axis at a time.
    #procedure.add_operation("Rotation", -orientation) #THIS LINE IS JUST A EXAMPLE, DONT RUN IT!!



    # O print abaixo é só para vocês verificarem o funcionamento, deve ser removido.
    #print("Viewpoint : position = {0}, orientation = {1}, fieldOfView = {2}".format(position, orientation, fieldOfView)) # imprime no terminal

def transform(translation, scale, rotation):
    """ Função usada para renderizar (na verdade coletar os dados) de Transform. """
    # A função transform será chamada quando se entrar em um nó X3D do tipo Transform
    # do grafo de cena. Os valores passados são a escala em um vetor [x, y, z]
    # indicando a escala em cada direção, a translação [x, y, z] nas respectivas
    # coordenadas e finalmente a rotação por [x, y, z, t] sendo definida pela rotação
    # do objeto ao redor do eixo x, y, z por t radianos, seguindo a regra da mão direita.
    # Quando se entrar em um nó transform se deverá salvar a matriz de transformação dos
    # modelos do mundo em alguma estrutura de pilha.

    # O print abaixo é só para vocês verificarem o funcionamento, deve ser removido.
    #print("Transform : ", end = '')
    if translation and not (translation[0] == 0 and translation[1] == 0  and translation[2] == 0):
        procedure.add_operation("Translation", translation)
        #print("translation = {0} ".format(translation), end = '') # imprime no terminal
    if scale and not (scale[0] == 1 and scale[1] == 1 and scale[2] == 1):
        procedure.add_operation("Scale", scale)
        #print("scale = {0} ".format(scale), end = '') # imprime no terminal
    if rotation and not (rotation[3] == 0):
        procedure.add_operation("Rotation", rotation)
        #print("rotation = {0} ".format(rotation), end = '') # imprime no terminal
    #print("")

def _transform():
    """ Função usada para renderizar (na verdade coletar os dados) de Transform. """
    # A função _transform será chamada quando se sair em um nó X3D do tipo Transform do
    # grafo de cena. Não são passados valores, porém quando se sai de um nó transform se
    # deverá recuperar a matriz de transformação dos modelos do mundo da estrutura de
    # pilha implementada.

    procedure.pop_operation()
    # O print abaixo é só para vocês verificarem o funcionamento, deve ser removido.
    #print("Saindo de Transform")

def triangleStripSet(point, stripCount, color):
    """ Função usada para renderizar TriangleStripSet. """
    # A função triangleStripSet é usada para desenhar tiras de triângulos interconectados,
    # você receberá as coordenadas dos pontos no parâmetro point, esses pontos são uma
    # lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x
    # do primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
    # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e assim
    # por diante. No TriangleStripSet a quantidade de vértices a serem usados é informado
    # em uma lista chamada stripCount (perceba que é uma lista).

    #triangleSet(point, color)
    # O print abaixo é só para vocês verificarem o funcionamento, deve ser removido.
    print("TriangleStripSet : pontos = {0} ".format(point), end = '') # imprime no terminal pontos
    for i, strip in enumerate(stripCount):
        print("strip[{0}] = {1} ".format(i, strip), end = '') # imprime no terminal
    print("")

def indexedTriangleStripSet(point, index, color):
    """ Função usada para renderizar IndexedTriangleStripSet. """
    # A função indexedTriangleStripSet é usada para desenhar tiras de triângulos
    # interconectados, você receberá as coordenadas dos pontos no parâmetro point, esses
    # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor
    # da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto, point[2]
    # o valor z da coordenada z do primeiro ponto. Já point[3] é a coordenada x do
    # segundo ponto e assim por diante. No IndexedTriangleStripSet uma lista informando
    # como conectar os vértices é informada em index, o valor -1 indica que a lista
    # acabou. A ordem de conexão será de 3 em 3 pulando um índice. Por exemplo: o
    # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
    # depois 2, 3 e 4, e assim por diante.
    
    
    print("-------------------indexedTriangleStripSet-----------------------------\n")
    pos = 0
    new_points = []
    print(point[36])

    while(index[pos+2] != -1):
        print("")
        print(pos)
        print("IndexedTriangleStripSet : pontos = {0}, index = {1}".format(point, index)) # imprime no terminal pontos
        print("tranformed : pontos = {0}".format(new_points)) # imprime no terminal pontos
        print("")

        for i in range(3):
            new_points.append(point[(int(index[pos+i])*3)])
            new_points.append(point[(int(index[pos+i])*3)+1])
            new_points.append(point[(int(index[pos+i])*3)+2])
        
        pos += 1

    
    
    
    # O print abaixo é só para vocês verificarem o funcionamento, deve ser removido.
    print("quantidade de triangulos:",len(new_points)/9)
    print("IndexedTriangleStripSet : pontos = {0}, index = {1}".format(point, index)) # imprime no terminal pontos
    print("tranformed : pontos = {0}".format(new_points)) # imprime no terminal pontos
    triangleSet(new_points, color)
    print("-------------------indexedTriangleStripSet-----------------------------\n")

def box(size, color):
    """ Função usada para renderizar Boxes. """
    # A função box é usada para desenhar paralelepípedos na cena. O Box é centrada no
    # (0, 0, 0) no sistema de coordenadas local e alinhado com os eixos de coordenadas
    # locais. O argumento size especifica as extensões da caixa ao longo dos eixos X, Y
    # e Z, respectivamente, e cada valor do tamanho deve ser maior que zero. Para desenha
    # essa /
    # *+caixa você vai provavelmente querer tesselar ela em triângulos, para isso
    # encontre os vértices e defina os triângulos.
    x = size[0] / 2
    y = size[1] / 2
    z = size[2] / 2

    #VERTICES
    A= [-x, y, -z]
    B= [x, y, -z]
    C= [-x, -y, -z]
    D= [x, -y, -z]
    E= [-x, y, z]
    F= [x, y, z]
    G= [-x, -y, z]
    H= [x, -y, z]
    #TRIANGLES
    Ts = [
        [A,B,C],
        [B,C,D],
        [B,F,D],
        [D,F,H],
        [C,G,D],
        [D,G,H],
        [A,C,E],
        [E,C,G],
        [A,E,B],
        [E,B,F],
        [E,F,G],
        [G,F,H] ]

    for t in Ts:
        print(t)
        triangleSet([ t[0][0], t[0][1], t[0][2], t[1][0], t[1][1], t[1][2], t[2][0], t[2][1], t[2][2] ], color)

    # O print abaixo é só para vocês verificarem o funcionamento, deve ser removido.
    #print("Box : size = {0}".format(size)) # imprime no terminal pontos


LARGURA = 30
ALTURA = 20

if __name__ == '__main__':

    # Valores padrão da aplicação
    width = LARGURA
    height = ALTURA
    x3d_file = "exemplo5.x3d"
    image_file = "tela.png"

    # Tratando entrada de parâmetro
    parser = argparse.ArgumentParser(add_help=False)   # parser para linha de comando
    parser.add_argument("-i", "--input", help="arquivo X3D de entrada")
    parser.add_argument("-o", "--output", help="arquivo 2D de saída (imagem)")
    parser.add_argument("-w", "--width", help="resolução horizonta", type=int)
    parser.add_argument("-h", "--height", help="resolução vertical", type=int)
    parser.add_argument("-q", "--quiet", help="não exibe janela de visualização", action='store_true')
    args = parser.parse_args() # parse the arguments
    if args.input: x3d_file = args.input
    if args.output: image_file = args.output
    if args.width: width = args.width
    if args.height: height = args.height

    # Iniciando simulação de GPU
    gpu.GPU(width, height, image_file)

    # Abre arquivo X3D
    scene = x3d.X3D(x3d_file)
    scene.set_resolution(width, height)

    # funções que irão fazer o rendering
    x3d.X3D.render["Polypoint2D"] = polypoint2D
    x3d.X3D.render["Polyline2D"] = polyline2D
    x3d.X3D.render["TriangleSet2D"] = triangleSet2D
    x3d.X3D.render["TriangleSet"] = triangleSet
    x3d.X3D.render["Viewpoint"] = viewpoint
    x3d.X3D.render["Transform"] = transform
    x3d.X3D.render["_Transform"] = _transform
    x3d.X3D.render["TriangleStripSet"] = triangleStripSet
    x3d.X3D.render["IndexedTriangleStripSet"] = indexedTriangleStripSet
    x3d.X3D.render["Box"] = box

    # Se no modo silencioso não configurar janela de visualização
    if not args.quiet:
        window = interface.Interface(width, height)
        scene.set_preview(window)

    scene.parse() # faz o traversal no grafo de cena

    # Se no modo silencioso salvar imagem e não mostrar janela de visualização
    if args.quiet:
        gpu.GPU.save_image() # Salva imagem em arquivo
    else:
        window.image_saver = gpu.GPU.save_image # pasa a função para salvar imagens
        window.preview(gpu.GPU._frame_buffer) # mostra janela de visualização
