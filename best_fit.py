#!/usr/bin/env python
# coding: utf-8

# author: Sanger Steel


import numpy as np
import numpy.polynomial.polynomial as npoly
from scipy.optimize import curve_fit, differential_evolution
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
import numpy.polynomial.polynomial as poly
import seaborn 




def compute_weighted_uncertainty(sys_error,error):
    weighted_uncertainty=np.maximum(sys_error,error)
    return weighted_uncertainty




def weights(compute_weighted_uncertainty, args):
    weights=1/(compute_weighted_uncertainty(*args))**2
    return weights




def segmentedRegression_1break(xData,yData):
        global piece1_params
        global ans_one
        global BIC_segReg_one
        global AIC_segReg_one
        def func(xVals,model_break,slopeA,slopeB,offsetA,offsetB):
            returnArray=[]
            for x in xVals:
                if x > model_break:
                    returnArray.append(slopeA * x + offsetA)
                else:
                    returnArray.append(slopeB * x + offsetB)


            return returnArray

        def sumSquaredError(parametersTuple):
            modely=func(xData,*parametersTuple)
            warnings.filterwarnings("ignore") # do not print warnings by genetic algorithm

            return np.sum((yData-modely)**2.0)

        def generate_genetic_Parameters():
            initial_parameters=[]
            x_max=np.max(xData)
            x_min=np.min(xData)
            y_max=np.max(yData)
            y_min=np.min(yData)
            slope=10*(y_max-y_min)/(x_max-x_min)

            initial_parameters.append([x_max,x_min])
            initial_parameters.append([-slope,slope])
            initial_parameters.append([-slope,slope])
            initial_parameters.append([y_max,y_min])
            initial_parameters.append([y_max,y_min])

            result=differential_evolution(sumSquaredError,initial_parameters,seed=3)

            return result.x

        geneticParameters = generate_genetic_Parameters()



        if (err_flag==1):
                piece1_params, pcov= curve_fit(func, xData, yData, geneticParameters,sigma=sigma) #Fits the data 
        if (err_flag==0):
                piece1_params, pcov= curve_fit(func, xData, yData, geneticParameters) #Fits the data 

        model=func(xData,*piece1_params)


        
        
        if (flag_BIC==1):
            BIC_segReg_one=compute_BIC(yData,model,5)
            return BIC_segReg_one
        if (flag_AIC==1):
            AIC_segReg_one=compute_AIC(yData,model,5)
            return AIC_segReg_one
        




def segmentedRegression_2break(xData,yData):
    global piece2_params
    global ans
    global BIC_segReg_two
    global AIC_segReg_two
    def func(xVals,break1,break2,slope1,offset1,slope_mid,offset_mid,slope2,offset2):
            returnArray=[]
            for x in xVals:
                if x < break1:
                    returnArray.append(slope1 * x + offset1)
                elif (np.logical_and(x >= break1,x<break2)):
                    returnArray.append(slope_mid * x + offset_mid)
                else:
                    returnArray.append(slope2 * x + offset2)

            return returnArray

    def sumSquaredError(parametersTuple): #Definition of an error function to minimize
        model_y=func(xData,*parametersTuple)
        warnings.filterwarnings("ignore") # Ignore warnings by genetic algorithm

        return np.sum((yData-model_y)**2.0)

    def generate_genetic_Parameters():
            initial_parameters=[]
            x_max=np.max(xData)
            x_min=np.min(xData)
            y_max=np.max(yData)
            y_min=np.min(yData)
            slope=10*(y_max-y_min)/(x_max-x_min)

            initial_parameters.append([x_max,x_min]) #Bounds for model break point
            initial_parameters.append([x_max,x_min])
            initial_parameters.append([-slope,slope]) 
            initial_parameters.append([-y_max,y_min]) 
            initial_parameters.append([-slope,slope]) 
            initial_parameters.append([-y_max,y_min]) 
            initial_parameters.append([-slope,slope])
            initial_parameters.append([y_max,y_min]) 



            result=differential_evolution(sumSquaredError,initial_parameters,seed=3)

            return result.x

    geneticParameters = generate_genetic_Parameters() #Generates genetic parameters


    if (err_flag==1):
            piece2_params, pcov= curve_fit(func, xData, yData, geneticParameters,sigma=sigma) #Fits the data 
    if (err_flag==0):
            piece2_params, pcov= curve_fit(func, xData, yData, geneticParameters) #Fits the data 




    model=func(xData,*piece2_params)

    if (flag_BIC==1):
        BIC_segReg_two=compute_BIC(yData,model,8)
        return BIC_segReg_two
    if (flag_AIC==1):
        AIC_segReg_two=compute_AIC(yData,model,8)
        return AIC_segReg_two
    
    
    




def segmentedRegression_3break(xData,yData):
    global piece3_params
    global ans_three
    global BIC_segReg_three
    global AIC_segReg_three
    def func(xVals,break1,break2,break3,slope1,offset1,slope2,offset2,slope3,offset3,slope4,offset4):
            returnArray=[]
            for x in xVals:
                if x < break1:
                    returnArray.append(slope1 * x + offset1)
                elif (np.logical_and(x >= break1,x<break2)):
                    returnArray.append(slope2 * x + offset2)
                elif (np.logical_and(x >= break2,x<break3)):
                    returnArray.append(slope3 * x + offset3)
                else:
                    returnArray.append(slope4 * x + offset4)

            return returnArray
   
    def sumSquaredError(parametersTuple): #Definition of an error function to minimize
        model_y=func(xData,*parametersTuple)
        warnings.filterwarnings("ignore") # Ignore warnings by genetic algorithm

        return np.sum((yData-model_y)**2.0)

    def generate_genetic_Parameters():
            initial_parameters=[]
            x_max=np.max(xData)
            x_min=np.min(xData)
            y_max=np.max(yData)
            y_min=np.min(yData)
            slope=10*(y_max-y_min)/(x_max-x_min)

            initial_parameters.append([x_max,x_min]) #Bounds for model break point
            initial_parameters.append([x_max,x_min])
            initial_parameters.append([x_max,x_min])
            initial_parameters.append([-slope,slope]) 
            initial_parameters.append([-y_max,y_min]) 
            initial_parameters.append([-slope,slope]) 
            initial_parameters.append([-y_max,y_min]) 
            initial_parameters.append([-slope,slope])
            initial_parameters.append([y_max,y_min]) 
            initial_parameters.append([-slope,slope])
            initial_parameters.append([y_max,y_min]) 



            result=differential_evolution(sumSquaredError,initial_parameters,seed=3)

            return result.x

    geneticParameters = generate_genetic_Parameters() #Generates genetic parameters


    if (err_flag==1):
            piece3_params, pcov= curve_fit(func, xData, yData, geneticParameters,sigma=sigma) #Fits the data 
    if (err_flag==0):
            piece3_params, pcov= curve_fit(func, xData, yData, geneticParameters) #Fits the data 

    model=func(xData,*piece3_params)

    if (flag_BIC==1):
        BIC_segReg_three=compute_BIC(yData,model,11)
        return BIC_segReg_three
    if (flag_AIC==1):
        AIC_segReg_three=compute_AIC(yData,model,11)
        return AIC_segReg_three
    
    
    





def compute_BIC(yData,model,variables):
    residual=yData-model
    SSE=np.sum(residual**2)
    return np.log(len(yData))*variables+len(yData)*np.log(SSE/len(yData))





def compute_AIC(yData,model,variables):
    residual=yData-model
    SSE=np.sum(residual**2)
    return 2*variables + len(yData) * np.log(SSE)





def p1(xData,a0,a1):
    return a0 + a1 * xData
def p2(xData,a0,a1,a2):
    return a0 + a1 * xData + a2 * xData**2
def p3(xData,a0,a1,a2,a3):
    return a0 + a1 * xData + a2 * xData**2 + a3 * xData**3
def p4(xData,a0,a1,a2,a3,a4):
    return a0 + a1 * xData + a2 * xData**2 + a3 * xData**3 + a4 * xData**4
def p5(xData,a0,a1,a2,a3,a4,a5):
    return a0 + a1 * xData + a2 * xData**2 + a3 * xData**3 + a4 * xData**4 + a5 * xData**5 
def p6(xData,a0,a1,a2,a3,a4,a5,a6):
    return a0 + a1 * xData + a2 * xData**2 + a3 * xData**3 + a4 * xData**4 + a5 * xData**5 + a6 * xData**6





def func_1break(xVals,model_break,slopeA,slopeB,offsetA,offsetB):
    returnArray=[]
    for x in xVals:
        if x > model_break:
            returnArray.append(slopeA * x + offsetA)
        else:
            returnArray.append(slopeB * x + offsetB)
    return returnArray

def func_2break(xVals,break1,break2,slope1,offset1,slope_mid,offset_mid,slope2,offset2):
    returnArray=[]
    for x in xVals:
        if x < break1:
            returnArray.append(slope1 * x + offset1)
        elif (np.logical_and(x >= break1,x<break2)):
            returnArray.append(slope_mid * x + offset_mid)
        else:
            returnArray.append(slope2 * x + offset2)

    return returnArray
        
def func_3break(xVals,break1,break2,break3,slope1,offset1,slope2,offset2,slope3,offset3,slope4,offset4):
            returnArray=[]
            for x in xVals:
                if x < break1:
                    returnArray.append(slope1 * x + offset1)
                elif (np.logical_and(x >= break1,x<break2)):
                    returnArray.append(slope2 * x + offset2)
                elif (np.logical_and(x >= break2,x<break3)):
                    returnArray.append(slope3 * x + offset3)
                else:
                    returnArray.append(slope4 * x + offset4)

            return returnArray





def polynom_best_fit(xData,yData):
        global polynom_params
        AICS=[]
        BICS=[]
        polynom_params=[]
        global best
        for i in [1,2,3,4,5]:
            params=[]
            
            params=poly.polyfit(xData,yData,i,w=weight)
            
            if len(params) == 2:
                model_p1=p1(xData,*params)
                if (flag_BIC==1):
                        BICS.append(compute_BIC(yData,model_p1,2))
                if (flag_AIC==1):
                        AICS.append(compute_AIC(yData,model_p1,2))
            if len(params) == 3:
                model_p2=p2(xData,*params)
                if (flag_BIC==1):
                        BICS.append(compute_BIC(yData,model_p1,3))
                if (flag_AIC==1):
                        AICS.append(compute_AIC(yData,model_p1,3))                                
            if len(params) == 4:
                model_p3=p3(xData,*params)
                if (flag_BIC==1):
                        BICS.append(compute_BIC(yData,model_p1,4))
                if (flag_AIC==1):
                        AICS.append(compute_AIC(yData,model_p1,4))
            if len(params) == 5:
                model_p4=p4(xData,*params)
                if (flag_BIC==1):
                        BICS.append(compute_BIC(yData,model_p1,5))
                if (flag_AIC==1):
                        AICS.append(compute_AIC(yData,model_p1,5))
            if len(params) == 6:
                model_p5=p5(xData,*params)
                if (flag_BIC==1):
                        BICS.append(compute_BIC(yData,model_p1,6))
                if (flag_AIC==1):
                        AICS.append(compute_AIC(yData,model_p1,6))
            else:
                continue
                
            def AIC_BIC_choose():
                if (flag_BIC==1):
                        best=np.where(BICS==min(BICS))
                        return best
                if (flag_AIC==1):
                        best=np.where(AICS==min(AICS))
                        return best
        
            best=AIC_BIC_choose()
            best_model=[]
            if best[0][0] == 0:
                if (flag_BIC==1):
                        polynom_params.append(tuple(poly.polyfit(xData,yData,1,w=weight)))
                        print('First degree best fit')
                        print('with BIC =', min(BICS))

                        return min(BICS)
                if (flag_AIC==1):
                        polynom_params.append(tuple(poly.polyfit(xData,yData,1,w=weight)))
                        print('First degree best fit')
                        print('with AIC =', min(AICS))

                        return min(AICS)
            if best[0][0] == 1:
                if (flag_BIC==1):
                        polynom_params.append(tuple(poly.polyfit(xData,yData,2,w=weight)))
                        print('Second degree best fit')
                        print('with BIC =', min(BICS))

                        return min(BICS)
                if (flag_AIC==1):
                        polynom_params.append(tuple(poly.polyfit(xData,yData,2,w=weight)))
                        print('Second degree best fit')
                        print('with AIC =', min(AICS))
                        
                        return min(AICS)
                
     
            if best[0][0] == 2:
                if (flag_BIC==1):
                        polynom_params.append(tuple(poly.polyfit(xData,yData,3,w=weight)))
                        print('Third degree best fit')
                        print('with BIC =', min(BICS))

                        return min(BICS)
                if (flag_AIC==1):
                        polynom_params.append(tuple(poly.polyfit(xData,yData,3,w=weight)))
                        print('Third degree best fit')
                        print('with AIC =', min(AICS))

                        return min(AICS)
            if best[0][0] == 3:
                if (flag_BIC==1):
                        polynom_params.append(tuple(poly.polyfit(xData,yData,4,w=weight)))
                        print('Fourth degree best fit')
                        print('with BIC =', min(BICS))

                        return min(BICS)
                if (flag_AIC==1):
                        polynom_params.append(tuple(poly.polyfit(xData,yData,4,w=weight)))
                        print('Fourth degree best fit')
                        print('with AIC =', min(AICS))

                        return min(AICS)
            if best[0][0] == 4:
                if (flag_BIC==1):
                        polynom_params.append(tuple(poly.polyfit(xData,yData,5,w=weight)))
                        print('Fifth degree best fit')
                        print('with BIC =', min(BICS))

                        return min(BICS)
                if (flag_AIC==1):
                        polynom_params.append(tuple(poly.polyfit(xData,yData,5,w=weight)))
                        print('Fifth degree best fit')
                        print('with AIC =', min(AICS))

                        return min(AICS)





def best_fit(x,y,method, error = None, **kwargs):
        """
        Applies polynomial fits of lowest degree 1 and highest degree
        5 and segmented regressions with at most 3 breakpoints in the
        model between x data and y data specified. Determines the best
        fit using Akaike or Bayesian Information Criteria. 
        
        
        ...
        
        
        Parameters
        ----------
        
        x : arraylike
               The independent variable. An array.
        
        y : arraylike
               The dependent variable. An array.
               
               
        method : string
               Either 'AIC' or 'BIC'. String input to determine the 
               choice of information criterion.
               
        
        error : arraylike, optional
               The uncertainty in the y data. An array of zeros
               with the same length as the data 
        
        
        sys_error : scalar, optional
               Systematic error for the data if present.
        
        Returns
        -------
        
        params : ndarray
        
               Fitting parameters of the optimal model. The exact
               significance of each parameter will be specified
               for segmented regressions, and it is assumed
               polynomial parameters are listed in the form
               [a0,a1,a2,...]
                         
               
        .. note:: x and y, and error must be of the same length.
                  The number of datapoints must exceed 8.
                  
                      
        """ 
    
        global flag_AIC
        global flag_BIC
        global sigma
        global weight
        global err_flag
        error=np.abs(error)
        fit_scores=[]
        sigma=np.zeros(len(y))
        weight=np.zeros(len(y))
        if error is None:
            err_flag=0
            for n in range(len(y)):
                sigma[n]= 0
                weight[n]= 1          

        flag_AIC=0
        flag_BIC=0
        sys_error = kwargs.get('sys_error', 0)

        if (method == 'AIC'):
            print('AIC will be used.')
            flag_AIC = 1
        if (method == 'BIC'):
            print('BIC will be used.')
            flag_BIC = 1
        if error is not None:
            err_flag=1
            error = np.asarray(error)
            for n in range(len(y)):
                sigma[n]=compute_weighted_uncertainty(sys_error,error[n])
                weight[n]=1/(sigma[n]**2)
        if len(y) > 3:
                if (3<len(y)<8):
                    raise ValueError('Length of x and y must at least exceed 8.')
                if (len(y) > 11):
                        print('Exploring different fits..')
                        print('Computing fit for segmented regression fit (one breakpoint)...')
                        segRegfit_one=segmentedRegression_1break(x,y)
                        print('Computing fit for segmented regression fit (two breakpoints)...')
                        segRegfit_two=segmentedRegression_2break(x,y)
                        print('Computing fit for segmented regression fit (three breakpoints)...')
                        segRegfit_three=segmentedRegression_3break(x,y)
                        print('Computing best polynomial fit...')
                        best_poly_fit = polynom_best_fit(x,y)
                        print('AIC/BIC for segmented regression fit (one breakpoint): ', segRegfit_one)
                        print('AIC/BIC for segmented regression fit (two breakpoints): ', segRegfit_two)
                        print('AIC/BIC for segmented regression fit (three breakpoint): ', segRegfit_three)
                        fit_scores.append(segRegfit_one)
                        fit_scores.append(segRegfit_two)
                        fit_scores.append(segRegfit_three)
                        fit_scores.append(best_poly_fit)
                        if np.where(fit_scores==min(fit_scores))[0][0] == 0:
                                print('Segmented regression (one breakpoint) is best fit with lowest' , method, 'value with the following parameters')
                                print('in the form [position of breakpoint, slope 1, slope 2, offset 1, offset 2].')
                                print(piece1_params)
                                return piece1_params
                        if np.where(fit_scores==min(fit_scores))[0][0] == 1:
                                print('Segmented regression (two breakpoints) is best fit with lowest' , method, 'value with the following parameters')
                                print('in the form [position of breakpoint 1, position of breakpoint 2, slope 1, offset 1, slope 2, offset 2, slope 3, offset 3].')
                                print(piece2_params)
                                return piece2_params
                        if np.where(fit_scores==min(fit_scores))[0][0] == 2:
                                print('Segmented regression (three breakpoints) is best fit with lowest' , method, 'value with the following parameters')
                                print('in the form [position of breakpoint 1, position of breakpoint 2, position of breakpoint 3, slope 1, offset 1, slope 2, offset 2, slope 3, offset 3, slope 4, offset 4].')
                                print(piece3_params)
                                return piece3_params
                        if np.where(fit_scores==min(fit_scores))[0][0] == 3:
                                print('Polynomial fit is best fit with lowest' , method, 'value.')
                                print(polynom_params)
                                return polynom_params 
                else: 
                        if (len(y) > 8):
                                print('Too few datapoints for segmented regression with three breakpoints.')
                                print('Exploring different fits..')
                                print('Computing fit for segmented regression fit (one breakpoint)...')
                                segRegfit_one=segmentedRegression_1break(x,y)
                                print('Computing fit for segmented regression fit (two breakpoints)...')
                                segRegfit_two=segmentedRegression_2break(x,y)
                                print('Computing best polynomial fit...')
                                best_poly_fit = polynom_best_fit(x,y)
                                print('AIC/BIC for segmented regression fit (one breakpoint): ', segRegfit_one)
                                print('AIC/BIC for segmented regression fit (two breakpoints): ', segRegfit_two)
                                fit_scores.append(segRegfit_one)
                                fit_scores.append(segRegfit_two)
                                fit_scores.append(best_poly_fit)
                                if np.where(fit_scores==min(fit_scores))[0][0] == 0:
                                        print('Segmented regression (one breakpoint) is best fit with lowest' , method, 'value with the following parameters')
                                        print('in the form [position of breakpoint, slope 1, slope 2, offset 1, offset 2].')
                                        print(piece1_params)
                                        return piece1_params
                                if np.where(fit_scores==min(fit_scores))[0][0] == 1:
                                        print('Segmented regression (two breakpoints) is best fit with lowest' , method, 'value with the following parameters')
                                        print('in the form [position of breakpoint 1, position of breakpoint 2, slope 1, offset 1, slope 2, offset 2, slope 3, offset 3].')
                                        print(piece2_params)
                                        return piece2_params

                                if np.where(fit_scores==min(fit_scores))[0][0] == 2:
                                        print('Polynomial fit is best fit with lowest' , method, 'value.')
                                        print(polynom_params)
                                        return polynom_params 
                       
                        
        else:
            raise ValueError('More parameters than datapoints for all fits.')







