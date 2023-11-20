function [ data_train, data_query ] = getData( MODE )
% Generate training and testing data

% Data Options:
%   1. Toy_Gaussian
%   2. Toy_Spiral
%   3. Toy_Circle
%   4. Caltech 101

showImg = 1; % Show training & testing images and their image feature vector (histogram representation)

PHOW_Sizes = [4 8 10]; % Multi-resolution, these values determine the scale of each layer.
PHOW_Step = 8; % The lower the denser. Select from {2,4,8,16}

switch MODE
    case 'Toy_Gaussian' % Gaussian distributed 2D points
        %rand('state', 0);
        %randn('state', 0);
        N= 150;
        D= 2;
        
        cov1 = randi(4);
        cov2 = randi(4);
        cov3 = randi(4);
        
        X1 = mgd(N, D, [randi(4)-1 randi(4)-1], [cov1 0;0 cov1]);
        X2 = mgd(N, D, [randi(4)-1 randi(4)-1], [cov2 0;0 cov2]);
        X3 = mgd(N, D, [randi(4)-1 randi(4)-1], [cov3 0;0 cov3]);
        
        X= real([X1; X2; X3]);
        X= bsxfun(@rdivide, bsxfun(@minus, X, mean(X)), var(X));
        Y= [ones(N, 1); ones(N, 1)*2; ones(N, 1)*3];
        
        data_train = [X Y];
        
    case 'Toy_Spiral' % Spiral (from Karpathy's matlab toolbox)
        
        N= 50;
        t = linspace(0.5, 2*pi, N);
        x = t.*cos(t);
        y = t.*sin(t);
        
        t = linspace(0.5, 2*pi, N);
        x2 = t.*cos(t+2);
        y2 = t.*sin(t+2);
        
        t = linspace(0.5, 2*pi, N);
        x3 = t.*cos(t+4);
        y3 = t.*sin(t+4);
        
        X= [[x' y']; [x2' y2']; [x3' y3']];
        X= bsxfun(@rdivide, bsxfun(@minus, X, mean(X)), var(X));
        Y= [ones(N, 1); ones(N, 1)*2; ones(N, 1)*3];
        
        data_train = [X Y];
        
    case 'Toy_Circle' % Circle
        
        N= 50;
        t = linspace(0, 2*pi, N);
        r = 0.4
        x = r*cos(t);
        y = r*sin(t);
        
        r = 0.8
        t = linspace(0, 2*pi, N);
        x2 = r*cos(t);
        y2 = r*sin(t);
        
        r = 1.2;
        t = linspace(0, 2*pi, N);
        x3 = r*cos(t);
        y3 = r*sin(t);
        
        X= [[x' y']; [x2' y2']; [x3' y3']];
        Y= [ones(N, 1); ones(N, 1)*2; ones(N, 1)*3];
        
        data_train = [X Y];
        
    case 'Caltech' % Caltech dataset
        close all;
        imgSel = [15 15]; % randomly select 15 images each class without replacement. (For both training & testing)
        folderName = './Caltech_101/101_ObjectCategories';
        classList = dir(folderName); % 폴더 목록을 가져옴->class name
        classList = {classList(3:end).name} % 10 classes
        
        disp('Loading training images...')
        % Load Images -> Description (Dense SIFT)
        cnt = 1;
        if showImg
            figure('Units','normalized','Position',[.05 .1 .4 .9]);
            suptitle('Training image samples');
        end
        for c = 1:length(classList)
            subFolderName = fullfile(folderName,classList{c}); % class c의 directory
            imgList = dir(fullfile(subFolderName,'*.jpg')); % class c의 모든 image list
            imgIdx{c} = randperm(length(imgList)); % 랜덤 인덱스
            imgIdx_tr = imgIdx{c}(1:imgSel(1)); % 앞의 15개 -> train set
            imgIdx_te = imgIdx{c}(imgSel(1)+1:sum(imgSel)); % 뒤의 15개 -> test set
            
            for i = 1:length(imgIdx_tr) 
                I = imread(fullfile(subFolderName,imgList(imgIdx_tr(i)).name));
                
                % Visualise
                if i < 6 & showImg
                    subaxis(length(classList),5,cnt,'SpacingVert',0,'MR',0);
                    imshow(I);
                    cnt = cnt+1;
                    drawnow;
                end
                
                if size(I,3) == 3 % RGB를 gray로 변환
                    I = rgb2gray(I); % PHOW work on gray scale image
                end
                
                % For details of image description, see http://www.vlfeat.org/matlab/vl_phow.html
                [~, desc_tr{c,i}] = vl_phow(single(I),'Sizes',PHOW_Sizes,'Step',PHOW_Step); %  extracts PHOW features (multi-scaled Dense SIFT)
            end
        end
        
        disp('Building visual codebook...')
        % Build visual vocabulary (codebook) for 'Bag-of-Words method'
        desc_sel = single(vl_colsubset(cat(2,desc_tr{:}), 10e4)); % Randomly select 100k SIFT descriptors for clustering
            % SIFT descriptor를 얻기 위한 코드
            % desc_sel : 10e4개의 descriptor
        
        % K-means clustering
        numBins = 4096; % for instance,
        
        
        %% write your own codes here
        tic
        % ...
        [~, centers] = kmeans(desc_sel, numBins, 'MaxIter', 1000, 'Start', 'plus');
        % ...
        toc

        disp('Encoding Images...')
        % Vector Quantisation
        
        %% write your own codes here
        % ...
        histogram_tr = zeros(length(classList), numBins); % Initialize histograms

        for c = 1:length(classList)
            subFolderName = fullfile(folderName, classList{c});
            imgList = dir(fullfile(subFolderName, '*.jpg'));
            imgIdx_tr = imgIdx{c}(1:imgSel(1)); % Training set indices
        
            for i = 1:length(imgIdx_tr)
                I = imread(fullfile(subFolderName, imgList(imgIdx_tr(i)).name));
                if size(I, 3) == 3
                    I = rgb2gray(I);
                end
                [~, desc_tr] = vl_phow(single(I), 'Sizes', PHOW_Sizes, 'Step', PHOW_Step);
        
                % Assign each descriptor to the nearest cluster center
                D = pdist2(single(desc_tr'), centers); % Compute distances to cluster centers
                [~, assignments] = min(D, [], 2); % Assign each descriptor to the nearest center
        
                % Build the histogram
                hist = histcounts(assignments, 1:numBins);
                histogram_tr(c, :) = histogram_tr(c, :) + hist'; % Sum up histograms for each class
            end
        end

        % histogram 정규화
        histogram_tr = bsxfun(@rdivide, histogram_tr, sum(histogram_tr, 2));
        % ...
        toc
        
        % Clear unused varibles to save memory
        clearvars desc_tr desc_sel
end

switch MODE
    case 'Caltech'
        if showImg
        figure('Units','normalized','Position',[.05 .1 .4 .9]);
        suptitle('Test image samples');
        end
        disp('Processing testing images...');
        cnt = 1;
        % Load Images -> Description (Dense SIFT)
        for c = 1:length(classList)
            subFolderName = fullfile(folderName,classList{c});
            imgList = dir(fullfile(subFolderName,'*.jpg'));
            
            for i = 1:length(imgIdx_te)
                I = imread(fullfile(subFolderName,imgList(imgIdx_te(i)).name));
                
                % Visualise
                if i < 6 & showImg
                    subaxis(length(classList),5,cnt,'SpacingVert',0,'MR',0);
                    imshow(I);
                    cnt = cnt+1;
                    drawnow;
                end
                
                if size(I,3) == 3
                    I = rgb2gray(I);
                end
                [~, desc_te{c,i}] = vl_phow(single(I),'Sizes',PHOW_Sizes,'Step',PHOW_Step);
            
            end
        end

        % Quantisation
        
        %% write your own codes here
        tic
        % ...
        histogram_te = zeros(length(classList), numBins); % Initialize histograms for test images
        for c = 1:length(classList)
            imgIdx_te = imgIdx{c}(imgSel(1)+1:sum(imgSel)); % Test set indices
        
            for i = 1:length(imgIdx_te)
                % Assign each descriptor to the nearest cluster center
                D = pdist2(single(desc_te'), centers);
                [~, assignments] = min(D, [], 2); % Assign each descriptor to the nearest center
        
                % Build the histogram
                hist = histcounts(assignments, 1:numBins);
                histogram_te(c, :) = histogram_te(c, :) + hist'; % Sum up histograms for each class
            end
        end
        
        histogram_te = bsxfun(@rdivide, histogram_te, sum(histogram_te, 2)); % histogram 정규화
        
        % ...
        toc
        

        %% Save the histogram data.
        label_train = ones(size(histogram_tr, 1), 1);
        label_query = ones(size(histogram_te, 1), 1);

        for i = 1:10 % label 설정
            label_train((i-1) * 15 + 1:i * 15) = i;
            label_query((i-1) * 15 + 1:i * 15) = i;
        end
        data_train = histogram_tr;
        data_query = histogram_te;
        
        %뒤에 label을 추가
        data_train(:,size(data_train,2)+1) = label_train;
        data_query(:,size(data_query,2)+1) = label_query;
    otherwise % Dense point for 2D toy data
        xrange = [-1.5 1.5];
        yrange = [-1.5 1.5];
        inc = 0.02;
        [x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));
        data_query = [x(:) y(:) zeros(length(x)^2,1)];
end
end

