function [outputArgs] = LSDecision(testData, LSPara, decisionCondition)
% To determine the class of the given testing dataset according the LS
% parameters;
% [outputArgs] = LSDecision(testData, LSPara, decisionCondition)
% each column is a data point;
% input variables:
%   testData: 
%   LSPara:
%   decisionCondition


% output variables (as a struct):
%   LS parameters: step
%   linearParameters: beta

% Author: CZ
% Version: 1.0
% Date              Status
% 26.09.2016        Draft
    
    if nargin < 2 || nargin > 3
        error('The number of inputs does not match!!');
    end;
    
    if nargin == 2
        defaultDecisionCondition = 0.5;
        decisionCondition = defaultDecisionCondition;
    end;
        
    if isempty(decisionCondition)
        decisionCondition = 0.5;
    end;
    
    outputArgs.decisionCondition = decisionCondition;
    
    estimatedClass = zeros(size(testData,2),1);
    
    for ii = 1:1:size(testData, 2)
        if (testData(:,ii)'*(LSPara)) >= decisionCondition
            estimatedClass(ii) = 1;
        else
            estimatedClass(ii) = 0;
        end;
    end;
    
    outputArgs.estimatedClass = estimatedClass;
end