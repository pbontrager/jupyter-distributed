import { IDebugger } from '@jupyterlab/debugger';
import { Widget } from '@lumino/widgets';

interface RankThread {
  rank: number;
  threadId: number;
}

/** Selects which stopped rank supplies the standard debugger models. */
export class DebuggerRankSelector extends Widget {
  constructor(debuggerService: IDebugger) {
    super({ node: Private.createNode() });
    this.title.label = 'Rank';
    this.title.caption = 'Select the distributed rank to inspect';
    this.addClass('jp-JupyterDistributedDebuggerRank');
    this._debugger = debuggerService;
    this._select = this.node.querySelector('select')!;
    this._select.addEventListener('change', this._onChange);
    debuggerService.eventMessage.connect(this._onDebugEvent, this);
    debuggerService.sessionChanged.connect(this._onSessionChanged, this);
    this.setHidden(true);
  }

  dispose(): void {
    if (this.isDisposed) {
      return;
    }
    this._select.removeEventListener('change', this._onChange);
    this._debugger.eventMessage.disconnect(this._onDebugEvent, this);
    this._debugger.sessionChanged.disconnect(this._onSessionChanged, this);
    super.dispose();
  }

  async selectRank(rank: number): Promise<{
    rank: number;
    availableRanks: number[];
  }> {
    if (!Number.isSafeInteger(rank) || rank < 0) {
      throw new Error('Rank must be a non-negative integer.');
    }
    if (!this._debugger.session?.isStarted) {
      throw new Error('Start a JupyterLab debugging session before selecting a rank.');
    }
    if (!this._threads.has(rank)) {
      await this._refresh();
    }
    if (!this._threads.has(rank)) {
      const available = [...this._threads.keys()].sort((a, b) => a - b);
      throw new Error(
        available.length > 0
          ? `Rank ${rank} is not stopped. Available ranks: ${available.join(', ')}.`
          : 'No distributed ranks are currently stopped.'
      );
    }
    if (!(await this._showRank(rank))) {
      throw new Error(`Unable to select stopped rank ${rank}.`);
    }
    return {
      rank: this._selectedRank,
      availableRanks: [...this._threads.keys()].sort((a, b) => a - b)
    };
  }

  private _onSessionChanged = (): void => {
    this._request += 1;
    this._threads.clear();
    this._replaceOptions([]);
    this.setHidden(true);
  };

  private _onDebugEvent(
    sender: IDebugger,
    event: IDebugger.ISession.Event
  ): void {
    if (event.event === 'stopped' || event.event === 'continued') {
      void this._refresh();
    } else if (event.event === 'terminated') {
      this._onSessionChanged();
    }
  }

  private async _refresh(): Promise<void> {
    const session = this._debugger.session;
    const request = ++this._request;
    if (!session?.isStarted) {
      return;
    }

    try {
      const info = await session.sendRequest('debugInfo', {});
      const threadsReply = await session.sendRequest('threads', {});
      if (request !== this._request || session !== this._debugger.session) {
        return;
      }

      const stopped = new Set(info.body.stoppedThreads);
      const rankThreads = threadsReply.body.threads
        .filter(thread => stopped.has(thread.id))
        .map(thread => Private.rankThread(thread.id, thread.name))
        .filter((item): item is RankThread => item !== null)
        .sort((a, b) => a.rank - b.rank);

      this._threads = new Map(
        rankThreads.map(item => [item.rank, item.threadId])
      );
      this._replaceOptions(rankThreads.map(item => item.rank));
      const visible = rankThreads.length > 0;
      this.setHidden(!visible);
      if (visible) {
        requestAnimationFrame(() => {
          this.parent?.fit();
          this.parent?.update();
        });
      }
      if (rankThreads.length === 0) {
        this._debugger.model.stoppedThreads.clear();
        this._debugger.model.callstack.frames = [];
        return;
      }

      const rank = this._threads.has(this._selectedRank)
        ? this._selectedRank
        : rankThreads[0].rank;
      await this._showRank(rank, request);
    } catch (error) {
      if (request === this._request) {
        console.warn('Unable to refresh distributed debugger ranks', error);
      }
    }
  }

  private async _showRank(
    rank: number,
    request = ++this._request
  ): Promise<boolean> {
    const session = this._debugger.session;
    const threadId = this._threads.get(rank);
    if (!session?.isStarted || threadId === undefined) {
      return false;
    }

    this._select.disabled = true;
    try {
      const reply = await session.sendRequest('stackTrace', { threadId });
      if (
        request !== this._request ||
        session !== this._debugger.session ||
        !reply.success
      ) {
        return false;
      }
      this._selectedRank = rank;
      this._select.value = String(rank);
      this._debugger.model.stoppedThreads = new Set([threadId]);
      this._debugger.model.callstack.frames = reply.body.stackFrames;
      return true;
    } catch (error) {
      if (request === this._request) {
        console.warn(`Unable to inspect distributed rank ${rank}`, error);
      }
      return false;
    } finally {
      if (request === this._request) {
        this._select.disabled = false;
      }
    }
  }

  private _replaceOptions(ranks: number[]): void {
    const values = Array.from(this._select.options, option =>
      Number(option.value)
    );
    if (
      values.length === ranks.length &&
      values.every((value, index) => value === ranks[index])
    ) {
      return;
    }
    this._select.replaceChildren(
      ...ranks.map(rank => {
        const option = document.createElement('option');
        option.value = String(rank);
        option.textContent = `Rank ${rank}`;
        return option;
      })
    );
  }

  private _onChange = (): void => {
    void this._showRank(Number(this._select.value));
  };

  private _debugger: IDebugger;
  private _request = 0;
  private _select: HTMLSelectElement;
  private _selectedRank = 0;
  private _threads = new Map<number, number>();
}

namespace Private {
  export function createNode(): HTMLElement {
    const label = document.createElement('label');
    const text = document.createElement('span');
    const select = document.createElement('select');
    text.textContent = 'Inspect:';
    select.setAttribute('aria-label', 'Distributed debugger rank');
    label.append(text, select);
    return label;
  }

  export function rankThread(
    threadId: number,
    name: string
  ): RankThread | null {
    const match = /^Rank (\d+)(?::|$)/.exec(name);
    if (!match) {
      return null;
    }
    const rank = Number(match[1]);
    return Number.isSafeInteger(rank) ? { rank, threadId } : null;
  }
}
