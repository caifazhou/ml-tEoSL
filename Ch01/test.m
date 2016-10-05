% function [] = test()

trainingData = dataGenerator(2,200,3);
trainX = cat(1,trainingData.generatedData{1,1}(:,1:2),trainingData.generatedData{1,2}(:,1:2))';
trainY = cat(1,trainingData.generatedData{1,1}(:,end),trainingData.generatedData{1,2}(:,end))';
testData = testDataGenerator(trainX, 0.05);
classColor{1,1} = 'g';
classColor{1,2} = 'r';

% Least sqaure
LSEstimatedPara = leastSquare(trainX, trainY);
LSEstimatedClass = LSDecision(testData.testDataset, LSEstimatedPara.beta, 0.5);

% kNN
KNNEstimatedClass = KNNEstimated(testData.testDataset, trainX, trainY, 10, 2);

% plot the training data
figure(1);
hold on
grid on
scatter(trainingData.generatedData{1,1}(:,1),trainingData.generatedData{1,1}(:,2),40,'filled','g','s')
scatter(trainingData.generatedData{1,2}(:,1),trainingData.generatedData{1,2}(:,2),40,'filled','r', 'd')
% scatter(testData.testDataset(1,:),testData.testDataset(2,:),20,'b','c');
% index0 = find(ones(size(LSEstimatedClass.estimatedClass, 1),1) - LSEstimatedClass.estimatedClass);
% index1 = find(LSEstimatedClass.estimatedClass);
% scatter(testData.testDataset(1,index0),testData.testDataset(2,index0),'MarkerFaceColor','g','MarkerEdgeColor','b');
% scatter(testData.testDataset(1,index1),testData.testDataset(2,index1),'MarkerFaceColor','r','MarkerEdgeColor','b');

% LS decision boundary
xSpace = linspace(floor(min(testData.testDataset(1,:))),ceil(max(testData.testDataset(1,:))));
ySpace = (LSEstimatedClass.decisionCondition - xSpace.*LSEstimatedPara.beta(1))./(LSEstimatedPara.beta(2));
plot(xSpace, ySpace,'-b','LineWidth', 2);

% kNN decision boundary
contour(testData.X,testData.Y, reshape(KNNEstimatedClass.estimatedClass,size(testData.X)),[0.5 0.5],'LineWidth', 2, 'LineColor','k');

% legendString = cell(1,2);
legendString{1,1} = 'Green (class 0)';
legendString{1,2} = 'Red (class 1)';
legendString{1,3} = ['Boundary (x^T\beta =',num2str(LSEstimatedClass.decisionCondition),')'];
legendString{1,4} = 'KNN Boundary';
legend(legendString, 'FontSize', 18);
hold off

% close all