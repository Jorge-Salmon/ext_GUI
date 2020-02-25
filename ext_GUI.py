from kivy.app import App
import numpy as np
from scipy.special import jv, yv
from scipy.integrate import trapz
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
import csv
import sys

wl=[]
n=[]
k=[]

with open('Au_k.csv', 'r') as file:
    my_reader = csv.reader(file, delimiter=',')
    for row in my_reader:
        wl.append(float(row[0]))
        k.append(float(row[1]))

with open('Au_n.csv', 'r') as file:
    my_r = csv.reader(file, delimiter=',')
    for row in my_r:
        n.append(float(row[1]))

wavel=[i*1000 for i in wl]
length=len(wl)

class MainPage(GridLayout):
    def __init__(self, **kwargs):
        super(MainPage, self).__init__(**kwargs)
        self.cols = 1
        self.padding=10

        self.inside = GridLayout() # Create a new grid layout
        self.inside.cols = 5 # set columns for the new grid layout
   
        self.add_widget(Label(text='Optical extinction of a Au nanosphere', 
          font_size= 20, bold=True, size_hint=(0.7, None), height=40))
        self.add_widget(self.inside)

        self.inside.add_widget(Label(text='Enter the diameter in nm (above 30):', bold=True, size_hint=(0.22, None), height=40))
        self.diameter = TextInput(multiline=False, size_hint = (0.05, None), height=40, input_filter='int')

        self.button = Button(text="Plot", size_hint=(0.15, None), bold=True, height=40)
        self.button.bind(on_press=self.plot)

        self.clear = Button(text="Clear", bold=True, size_hint=(0.15, None), height=40)
        self.clear.bind(on_press=self.Clear)

        self.inside.add_widget(self.diameter)
        self.inside.add_widget(self.button)
        self.inside.add_widget(self.clear)


    def MieQ(self, m, wavelength, diameter, nMedium=1.0, asDict=False, asCrossSection=False):
#  http://pymiescatt.readthedocs.io/en/latest/forward.html#MieQ
        nMedium = nMedium.real
        m /= nMedium
        wavelength /= nMedium
        x = np.pi*diameter/wavelength
        if x==0:
          return 0, 0, 0, 1.5, 0, 0, 0
        elif x<=0.05:
          return RayleighMieQ(m, wavelength, diameter, nMedium, asDict)
        elif x>0.05:
          nmax = np.round(2+x+4*(x**(1/3)))
          n = np.arange(1,nmax+1)
          n1 = 2*n+1
          n2 = n*(n+2)/(n+1)
          n3 = n1/(n*(n+1))
          x2 = x**2

          an,bn = self.Mie_ab(m,x)

          qext = (2/x2)*np.sum(n1*(an.real+bn.real))
          qsca = (2/x2)*np.sum(n1*(an.real**2+an.imag**2+bn.real**2+bn.imag**2))
          qabs = qext-qsca

          g1 = [an.real[1:int(nmax)],
               an.imag[1:int(nmax)],
               bn.real[1:int(nmax)],
               bn.imag[1:int(nmax)]]
          g1 = [np.append(x, 0.0) for x in g1]
          g = (4/(qsca*x2))*np.sum((n2*(an.real*g1[0]+an.imag*g1[1]+bn.real*g1[2]+bn.imag*g1[3]))+(n3*(an.real*bn.real+an.imag*bn.imag)))
   
          qpr = qext-qsca*g
          qback = (1/x2)*(np.abs(np.sum(n1*((-1)**n)*(an-bn)))**2)
          qratio = qback/qsca
          if asCrossSection:
            css = np.pi*(diameter/2)**2
            cext = css*qext
            csca = css*qsca
            cabs = css*qabs
            cpr = css*qpr
            cback = css*qback
            cratio = css*qratio
            if asDict:
              return dict(Cext=cext,Csca=csca,Cabs=cabs,g=g,Cpr=cpr,Cback=cback,Cratio=cratio)
            else:
              return cext#, csca, cabs, g, cpr, cback, cratio
          else:
            if asDict:
              return dict(Qext=qext,Qsca=qsca,Qabs=qabs,g=g,Qpr=qpr,Qback=qback,Qratio=qratio)
            else:
              return qext#, qsca, qabs, g, qpr, qback, qratio

    def Mie_ab(self, m,x):
#  http://pymiescatt.readthedocs.io/en/latest/forward.html#Mie_ab
        mx = m*x
        nmax = np.round(2+x+4*(x**(1/3)))
        nmx = np.round(max(nmax,np.abs(mx))+16)
        n = np.arange(1,nmax+1) #
        nu = n + 0.5 #

        sx = np.sqrt(0.5*np.pi*x)
 
        px = sx*jv(nu,x) #
        p1x = np.append(np.sin(x), px[0:int(nmax)-1]) #

        chx = -sx*yv(nu,x) #
        ch1x = np.append(np.cos(x), chx[0:int(nmax)-1]) #
  
        gsx = px-(0+1j)*chx #
        gs1x = p1x-(0+1j)*ch1x #

  # B&H Equation 4.89
        Dn = np.zeros(int(nmx),dtype=complex)
        for i in range(int(nmx)-1,1,-1):
          Dn[i-1] = (i/mx)-(1/(Dn[i]+i/mx))

        D = Dn[1:int(nmax)+1] # Dn(mx), drop terms beyond nMax
        da = D/m+n/x
        db = m*D+n/x

        an = (da*px-p1x)/(da*gsx-gs1x)
        bn = (db*px-p1x)/(db*gsx-gs1x)

        return an, bn

    def RayleighMieQ(self, m, wavelength, diameter, nMedium=1.0, asDict=False, asCrossSection=False):
#  http://pymiescatt.readthedocs.io/en/latest/forward.html#RayleighMieQ
        nMedium = nMedium.real
        m /= nMedium
        wavelength /= nMedium
        x = np.pi*diameter/wavelength
        if x==0:
          return 0, 0, 0, 1.5, 0, 0, 0
        elif x>0:
          LL = (m**2-1)/(m**2+2) # Lorentz-Lorenz term
          LLabsSq = np.abs(LL)**2
          qsca = 8*LLabsSq*(x**4)/3 # B&H eq 5.8
          qabs=4*x*LL.imag # B&H eq. 5.11
          qext=qsca+qabs
          qback = 1.5*qsca # B&H eq. 5.9
          qratio = 1.5
          g = 0
          qpr = qext
          if asCrossSection:
            css = np.pi*(diameter/2)**2
            cext = css*qext
            csca = css*qsca
            cabs = css*qabs
            cpr = css*qpr
            cback = css*qback
            cratio = css*qratio
            if asDict:
              return dict(Cext=cext,Csca=csca,Cabs=cabs,g=g,Cpr=cpr,Cback=cback,Cratio=cratio)
            else:
              return cext, csca, cabs, g, cpr, cback, cratio
          else:
            if asDict:
              return dict(Qext=qext,Qsca=qsca,Qabs=qabs,g=g,Qpr=qpr,Qback=qback,Qratio=qratio)
            else:
              return qext, qsca, qabs, g, qpr, qback, qratio

# Loading Au refractive index function

    def plot(self, instance):
        plt.clf()
        lista=[]
        d=int(self.diameter.text)
        
        for i in range(len(wl)):
            Qext=self.MieQ(complex(n[i], k[i]), wavel[i], d, asDict=False)
            lista.append([Qext])

        fig, ax = plt.subplots()

        ax.plot(wavel, lista, 'o', color='m')
        ax.set_xlim(400, 1200)  # decreasing time
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Optical extinction (a. u.)')
        ax.set_title('Mie theory')
        ax.grid(True)

        self.oras=FigureCanvasKivyAgg(plt.gcf(), size_hint=(1.0, 10))
        self.add_widget(self.oras)

        self.diameter.text=''

    def Clear(self, instance):
        self.remove_widget(self.oras)

class App(App):
    def build(self):
        return MainPage()


if __name__ == '__main__':
    App().run()
