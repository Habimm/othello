(function (O) {
  'use strict';

  // UI {{{1

  function drawGameBoard(board, player, moves) {
  var ss = [];
  var attackable = [];
  moves.forEach(function (m) {
    if (!m.isPassingMove)
    attackable[O.ix(m.x, m.y)] = true;
  });

  ss.push('<table>');
  for (var y = -1; y < O.N; y++) {
    ss.push('<tr>');
    for (var x = -1; x < O.N; x++) {
    if (0 <= y && 0 <= x) {
      ss.push('<td class="');
      ss.push('cell');
      ss.push(' ');
      ss.push(attackable[O.ix(x, y)] ? player : board[O.ix(x, y)]);
      ss.push(' ');
      ss.push(attackable[O.ix(x, y)] ? 'attackable' : '');
      ss.push('" id="');
      ss.push('cell_' + x + '_' + y);
      ss.push('">');
      ss.push('<span class="disc"></span>');
      ss.push('</td>');
    } else if (0 <= x && y === -1) {
      ss.push('<th>' + String.fromCharCode('a'.charCodeAt(0)+x) + '</th>');
    } else if (x === -1 && 0 <= y) {
      ss.push('<th>' + (y + 1) + '</th>');
    } else /* if (x === -1 && y === -1) */ {
      ss.push('<th></th>');
    }
    }
    ss.push('</tr>');
  }
  ss.push('</table>');

  $('#game-board').html(ss.join(''));
  $('#current-player-name').text(player);
  }

  function resetUI() {
  $('#console').empty();
  $('#message').empty();
  }

  function setUpUIToChooseMove(gameTree) {
  $('#message').text('Choose your move.');
  gameTree.moves.forEach(function (m, i) {
    if (m.isPassingMove) {
    $('#console').append(
      $('<input type="button" class="btn">')
      .val(O.nameMove(m))
      .click(function () {
      shiftToNewGameTree(O.force(m.gameTreePromise));
      })
    );
    } else {
    $('#cell_' + m.x + '_' + m.y)
    .click(function () {
      shiftToNewGameTree(O.force(m.gameTreePromise));
    });
    }
  });
  }

  function setUpUIToReset() {
  resetGame();
  if ($('#repeat-games:checked').length)
    startNewGame();
  }

  var minimumDelayForAI = 500;  // milliseconds

  function sendStateToServer(data) {
    fetch('http://127.0.0.1:5000/receive_json', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
      console.log("Server response:", data);
    })
    .catch(error => {
      console.error("Error sending data to server:", error);
    });
  }

  function chooseMoveByAI(gameTree, ai) {
    $('#message').text('Now thinking...');
    setTimeout(
      function () {
        // Convert the current Othello state to the desired format
        const formattedState = formatOthelloState(gameTree.board);

        // Sending the formatted Othello state to the server
        sendStateToServer({ "Current Othello state": formattedState });

        // Printing the formatted Othello board state
        console.log("Current Othello state:", formattedState);

        // Printing the list of legal moves
        console.log("Legal moves:", gameTree.moves);

        // Assertion: There must always be legal moves available
        if (gameTree.moves.length === 0) {
          throw new Error("Assertion failed: No legal moves available");
        }

        const firstMove = gameTree.moves[0];
        console.log("First legal move:", [firstMove.x, firstMove.y]);

        // Fetch the Othello state if that move is taken
        const newGameTree = O.force(firstMove.gameTreePromise);

        setTimeout(
          function () {
            shiftToNewGameTree(newGameTree);
          },
          Math.max(minimumDelayForAI - 0, 1)
        );
      },
      1
    );
  }

  function formatOthelloState(state) {
  const stateMapping = {
    'empty': 0,
    'black': 1,
    'white': 2
  };

  // Convert the state string to its corresponding numerical value
  const numericState = state.map(item => stateMapping[item]);

  // Split the state into 8x8 chunks
  let formattedState = [];
  for (let i = 0; i < numericState.length; i += 8) {
    formattedState.push(numericState.slice(i, i + 8));
  }

  return formattedState;
  }

  function showWinner(board) {
  var r = O.judge(board);
  $('#message').text(
    r === 0 ?
    'The game ends in a draw.' :
    'The winner is ' + (r === 1 ? O.BLACK : O.WHITE) + '.'
  );
  }

  var playerTable = {};

  function makePlayer(playerType) {
  if (playerType === 'human') {
    return setUpUIToChooseMove;
  } else {
    var ai = O.makeAI(playerType);
    return function (gameTree) {
    chooseMoveByAI(gameTree, ai);
    };
  }
  }

  function blackPlayerType() {
  return $('#black-player-type').val();
  }

  function whitePlayerType() {
  return $('#white-player-type').val();
  }

  function swapPlayerTypes() {
  var t = $('#black-player-type').val();
  $('#black-player-type').val($('#white-player-type').val()).change();
  $('#white-player-type').val(t).change();
  }

  function shiftToNewGameTree(gameTree) {
  drawGameBoard(gameTree.board, gameTree.player, gameTree.moves);
  resetUI();
  if (gameTree.moves.length === 0) {
    showWinner(gameTree.board);
    recordStat(gameTree.board);
    if ($('#repeat-games:checked').length)
    showStat();
    setUpUIToReset();
  } else {
    playerTable[gameTree.player](gameTree);
  }
  }

  var stats = {};

  function recordStat(board) {
  var s = stats[[blackPlayerType(), whitePlayerType()]] || {b: 0, w: 0, d: 0};
  var r = O.judge(board);
  if (r === 1)
    s.b++;
  if (r === 0)
    s.d++;
  if (r === -1)
    s.w++;
  stats[[blackPlayerType(), whitePlayerType()]] = s;
  }

  function showStat() {
  var s = stats[[blackPlayerType(), whitePlayerType()]];
  $('#stats').text('Black: ' + s.b + ', White: ' + s.w + ', Draw: ' + s.d);
  }

  function resetGame() {
  $('#preference-pane :input:not(#repeat-games)')
    .removeClass('disabled')
    .removeAttr('disabled');
  }

  function startNewGame() {
  $('#preference-pane :input:not(#repeat-games)')
    .addClass('disabled')
    .attr('disabled', 'disabled');
  playerTable[O.BLACK] = makePlayer(blackPlayerType());
  playerTable[O.WHITE] = makePlayer(whitePlayerType());
  shiftToNewGameTree(O.makeInitialGameTree());
  }




  // Startup {{{1

  $('#start-button').click(function () {startNewGame();});
  $('#add-new-ai-button').click(function () {O.addNewAI();});
  $('#swap-player-types-button').click(function () {swapPlayerTypes();});
  resetGame();
  drawGameBoard(O.makeInitialGameBoard(), '-', []);




  //}}}
})(othello);
// vim: expandtab softtabstop=2 shiftwidth=2 foldmethod=marker
