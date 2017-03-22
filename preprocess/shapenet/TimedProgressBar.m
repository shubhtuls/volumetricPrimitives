classdef TimedProgressBar < handle
    %PROGRESSBAR Progress bar class for matlab loops which also works with
    %   parfor. PROGRESSBAR works by creating a transient file in your working
    %   directory, and then keeping track of the loop's progress within that
    %   file. This workaround is necessary because parfor workers cannot
    %   communicate with one another so there is no simple way to know which
    %   iterations have finished and which haven't.
    %   Meanwhile, refrain any other competing output to the command line,
    %   otherwise it will mess it. Just use the input strings to comunicate
    %   while de cycle is going on.
    %
    % METHODS:  TimedProgressBar(); constructs an object
    %               and initializes the progress monitor 
    %               for a set of targetCount upcoming calculations.
    %           progress(); updates the progress inside your loop and
    %               displays an updated progress bar.
    %           stop(); deletes progressbar_(random_number).txt and finalizes
    %               the progress bar.
    %
    % EXAMPLE: 
    %           targetCount = 100;
    %           barWidth= targetCount/2;
    %           p = TimedProgressBar( targetCount, barWidth, ...
    %                                 'Computing, please wait for ', ...
    %                                 ', already completed ', ...
    %                                 'Concluded in ' );
    %           parfor i=1:targetCount
    %              pause(rand);     % Replace with real code
    %              p.progress;      % Also percent = p.progress;
    %           end
    %           p.stop;             % Also percent = p.stop;
    %
    % To get percentage numbers from progress and stop methods call them like:
    %       percent = p.progress;
    %       percent = p.stop;
    %
    % date: 2014/05/27
    % author: Antonio Jose Cacho, ajsccacho@gmailcom
    % author: Stefan Doerr (previous version: ProgressBar)
    % author: Jeremy Scheff (previous version: parfor_progress)

    properties ( SetAccess= protected, GetAccess= protected )
        fname
        format
        waitMsg
        percentDoneMsg
        finishTimeMsg
        barWidth
        textWidth
        rewindLength
    end
    
    methods
        
        function obj= TimedProgressBar( targetCount, barWidth, waitMsg, ...
                                        percentDoneMsg, finishTimeMsg )
            mkdirOptional('./.timedBar');
            obj.fname= obj.setFName();
            f = fopen(obj.fname, 'w');
            if f<0
                error('Do you have write permissions for %s?', pwd);
            end
            ticInit= tic;
            fprintf(f, '%ld\n', ticInit );    % Save the initial tic mark at the top of progress.txt
            fprintf(f, '%d\n', targetCount);  % Save targetCount at the 2nd line of progress.txt
            fclose(f);
            
            obj.barWidth= barWidth;           % Width of progress bar
            obj.percentDoneMsg= percentDoneMsg;
            waitSz= length( waitMsg );
            finishSz= length( finishTimeMsg );
            if waitSz > finishSz
                obj.waitMsg= waitMsg;
                obj.finishTimeMsg= [ repmat( ' ', 1, waitSz - finishSz ), ...
                    finishTimeMsg ];
            else
                obj.finishTimeMsg= finishTimeMsg;
                obj.waitMsg= [ repmat( ' ', 1, finishSz - waitSz ), ...
                    waitMsg ];
            end
            obj.format= [ '%03d:%02d:%04.1f' ...         % hh:mm:ss.s
                          obj.percentDoneMsg '%3.0f%%']; % 4 characters wide, percentage
            obj.textWidth= length(obj.waitMsg) + 3 + 1 + 2 + 1 + 4 + ...
                           length(obj.percentDoneMsg) + 1 + 2 + 1;
            fprintf(1, [  repmat( ' ', 1, 20 ) '\n' ] );  % buffers multitasking uncertanties of the output to the command line
            obj.showStatus( 0, [ repmat( ' ', 1, obj.textWidth-4 ),'  0%' ]  );
        end
        
        function percent= progress(obj)
            obj.updateFile();
            [ percent, remainingTime, timeElapsed ]= obj.getProgress();
            timedMsg= obj.getTimedMsg( remainingTime, percent );
            obj.showStatus( percent, [ obj.waitMsg timedMsg ] );
        end
        
        function [ percent, timeElapsed ]= stop(obj)
            [ percent, ~, timeElapsed ]= obj.getProgress();
            delete( obj.fname );
            percent= 100;
            timedMsg= obj.getTimedMsg( timeElapsed, percent );
            obj.showStatus( percent, [ obj.finishTimeMsg timedMsg ] );
        end
        
    end
        
    methods  ( Static )
        
        function fname= getFName()
            baseName= './.timedBar/timedProgressbar_Transient(deleteThis)_';
            fname= [  baseName, num2str(randi(1000)) '.txt' ];
            while exist( fname, 'file' )
                fname= [ baseName, num2str(randi(1000)) '.txt' ];
            end
        end
        
    end
        
        
    methods  ( Access= protected )
        
        function fname= setFName(obj)
            fname= obj.getFName();
            while exist( fname, 'file' )
                fname= obj.getFName();
            end
        end
        
        function [ percent, remainingTime, timeElapsed ]= getProgress(obj)
            f= fopen( obj.fname, 'r' );
            [ readbuffer, progressCountPlus2 ]= fscanf( f, '%ld' );
            fclose( f );
            
            ticInit= uint64( readbuffer(1) );
            timeElapsed= toc( ticInit );
            
            progressCount= progressCountPlus2 - 2;
            targetCount= double( readbuffer(2) );       % avoids Matlab's buggy assumption of readbuffer type as uint64
            
            percent = progressCount * 100 / targetCount;
            if percent > 0
                remainingTime= timeElapsed * ( 100 / percent - 1 );
            else
                remainingTime= timeElapsed * 200;   % initial duration estimate
            end
        end
        
        function showStatus( obj, percent, statusMsg )
            x= round( percent * obj.barWidth / 100 );
            marker= '>';
            if x < obj.barWidth
                bar= [ ' [', repmat( '=', 1, x ), marker, ...
                             repmat( ' ', 1, obj.barWidth - x - 1 ), ']' ];
            else
                bar= [ ' [', repmat( '=', 1, x ), ...
                             repmat( ' ', 1, obj.barWidth - x ), ']' ];
            end
            statusLine= [ statusMsg, bar ];

            % console print:
            % <--------status message----------->_[<------progress bar----->]
            % Wait for 001:28:36.7, completed 38% [========>                ]
            
            if percent > 0 && percent < 100
                    cursorRewinder= repmat( char(8), 1, 1+obj.rewindLength );
                    disp( [ cursorRewinder, statusLine ] );
            elseif percent == 100
                    cursorRewinder= repmat( char(8), 1, 1+obj.rewindLength );
                    disp( [ cursorRewinder, statusLine ] );
            else    % percent == 0
                disp( statusLine );
            end
            obj.rewindLength= length( statusLine );
        end
        
        function timedMsg= getTimedMsg( obj, timeSec, percent )
            hh= fix( timeSec / 3600 );
            mm= fix( ( timeSec - hh * 3600 ) / 60 );
            ss= max( ( timeSec - hh * 3600 - mm * 60 ), 0.1 );
            timedMsg = sprintf( obj.format, hh, mm, ss, percent );
        end
        
        function updateFile(obj)
            if ~exist( obj.fname, 'file' )
                error( [ obj.fname ' not found. It must have been deleted.' ] );
            end
            f = fopen(obj.fname, 'a');
            fprintf(f, '1\n');
            fclose(f);
        end
        
    end
    
end
