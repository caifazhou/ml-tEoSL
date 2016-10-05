function [ outputArgs ] = dataGenerator( numberOfClass, numberOfExample, typeOfScenario)
% DATAGENERATOR To generate the data for simulation
% [ outputArgs ] = dataGenerator( numberOfClass, numberOfExample, typeOfScenario)
% range of data: [0 2;0 2]
% minimum step: 0.01
% Scenario 1: training data in each class were generated from bivariate Gaussain distribution 
% with uncorrelated components and different means;
% Scenario 2: The training data in each class came from a mixture of 10
% low-variance Gaussina distributon, with individual means themselves
% distributed as Gaussian
% Scenario 3: A scenario between 1 and 2 but closer to scenario 2
% input variables:
%   numberOfClass: 
%   numberOfExample:
%   typeOfScenario:
% defaultNumberOfClass = 2;
% defaultNumberOfExample = 100;
% defaultTypeOfScenario = 1;
% output variables (as a struct):
%     generatedData
%     inputParameters

% Author: CZ
% Version: 1.0
% Date              Status
% 25.09.2016        Draft
% 26.09.2016        add scenario 2

    if nargin > 3
        error('Number of inputs does not match!');
    end;
    if nargin == 0
        defaultNumberOfClass = 2;
        defaultNumberOfExample = 100;
        defaultTypeOfScenario = 1;
        numberOfClass = defaultNumberOfClass;
        numberOfExample = defaultNumberOfExample;
        typeOfScenario = defaultTypeOfScenario;
    end;

    if nargin == 1
        defaultNumberOfExample = 100;
        defaultTypeOfScenario = 1;
        numberOfExample = defaultNumberOfExample;
        typeOfScenario = defaultTypeOfScenario;
    end;
    if nargin == 2
        defaultTypeOfScenario = 1;
        typeOfScenario = defaultTypeOfScenario;
    end;


    outputArgs.inputPara.numberOfClass = numberOfClass;
    outputArgs.inputPara.numberOfExample = numberOfExample;
    outputArgs.inputPara.typeOfScenario = typeOfScenario;
    generatedData = cell(1, numberOfClass);

    % % for scenario 1
    if typeOfScenario == 1
        mu = cell(1,2);
        mu{1,1} = [0.2 0.2];
        mu{1,2} = [1.8 1.8];
        sigma = eye(2);
        rng('shuffle');
        for ii =1:1:numberOfClass
            generatedData{1,ii} = cat(2,mvnrnd(mu{1,ii}',sigma./2, numberOfExample),(ii-1).*ones(numberOfExample,1)); 
        end;
        outputArgs.generatedData = generatedData;
    end;

% %     for scenario 2
    if typeOfScenario == 2
        rng('shuffle');
        numberOfComponents = 10;
        sigma = eye(2);
        
        weights = ones(1,numberOfComponents)./numberOfComponents;
        
        gmmSigma = cell(1, numberOfComponents);
        
        for ii =1:1:numberOfComponents
            gmmSigma{1, ii} = [0.1,0.05;0.05,0.1];
        end;
        
        for ii = 1:1:numberOfClass
            mu = mvnrnd([1 1]',sigma,10);
            gmm = gmdistribution(mu, cat(3, gmmSigma{:}), weights);
            generatedData{1,ii} = cat(2,random(gmm, numberOfExample), (ii-1).*ones(numberOfExample,1));
        end;
        outputArgs.generatedData = generatedData;
    end;
    % % for scenario 3
    if typeOfScenario == 3
        rng('shuffle');
        for ii =1:1:numberOfClass
            sigma = eye(2);
            muTemp = mvnrnd([1 0]',sigma,10);
            mu = muTemp(randi(10,1,1),:);
            generatedData{1,ii} = cat(2, mvnrnd(mu', sigma./5, numberOfExample),(ii-1).*ones(numberOfExample,1));
        end;
        outputArgs.generatedData = generatedData;
    end;
end

