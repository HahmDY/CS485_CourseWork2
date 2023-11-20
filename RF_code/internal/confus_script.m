[~,c] = max(p_rf');
accuracy_rf = sum(c==data_test(:,end)')/length(c); % Classification accuracy (for Caltech dataset)
idx = sub2ind([10, 10], data_test(:,end)', c) ;
conf = zeros(10) ;
conf = vl_binsum(conf, ones(size(idx)), idx) ;

figure;
subplot(1,2,1);
imagesc(conf);
title(sprintf('Confusion matrix (%.2f %% accuracy)', 100 * accuracy_rf) ) ;
subplot(1,2,2);
confusionchart(confusionmat(data_test(:,end)',c));