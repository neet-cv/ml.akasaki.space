import React, { useEffect, useState } from "react";
import Layout from "@theme/Layout";
import clsx from "clsx";
import arrayShuffle from "array-shuffle";

function About() {
  return (
    <Layout>
      <Friends />
      <p style={{ paddingLeft: "20px" }}>
        The list is random. try to refresh the page.
      </p>
    </Layout>
  );
}

interface FriendData {
  pic: string;
  name: string;
  intro: string;
  url: string;
  note: string;
}

function githubPic(name: string) {
  return `https://github.yuuza.net/${name}.png`;
}

var friendsData: FriendData[] = [
  {
    pic: githubPic("visualDust"),
    name: "Gavin Gong",
    intro:
      "Rubbish CVer | Poor LaTex speaker | Half stack developer | 键圈躺尸砖家",
    url: "https://focus.akasaki.space/",
    note: "Project Launcher & Maintainer, focusing on applied deep learning techniques. Also know him as VisualDust or MiyaAkasaki.",
  },
  {
    pic: githubPic("lideming"),
    name: "lideming",
    intro: "Building random things with Deno, Node and .NET Core.",
    url: "https://yuuza.net/",
    note: "Help with frontends since the project maintainer is new to frontends.",
  },
  {
    pic: githubPic("papercube"),
    name: "PaperCube",
    intro: "一个正在学习原理内容的软件开发者，睡大觉能手和前•算法竞赛生",
    url: "https://github.com/papercube",
    note: "Sometimes helps with multi language translations.",
  },
  {
    pic: githubPic("Therainisme"),
    name: "Therainisme",
    intro: "寄忆犹新",
    url: "https://notebook.therainisme.com/",
    note: "Used to help us with frontend issues.",
  },
  {
    pic: githubPic("AndSonder"),
    name: "Sonder",
    intro: "life is but a span, I use python",
    url: "https://blog.keter.top/",
    note: "Focusing on neural network adversial attacks and related things.",
  },
  {
    pic: githubPic("Zerorains"),
    name: "Zerorains",
    intro: "life is but a span, I use python",
    url: "blog.zerorains.top",
    note: "Focusing on semantic segmentation and image matting.",
  },
  {
    pic: githubPic("PommesPeter"),
    name: "PommesPeter",
    intro: "I want to be strong. But it seems so hard.",
    url: "https://blog.pommespeter.com/",
    note: "Focusing on low-light image enhancement and image processing.",
  },
  {
    pic: "https://xiaomai-aliyunoss.oss-cn-shenzhen.aliyuncs.com/img/20220116171846.jpg",
    name: "RuoMengAwA",
    intro: "一个喜欢摸鱼的菜狗，目前主要做低照度增强方向的研究",
    url: "https://github.com/RuoMengAwA",
    note: "Focusing on low-light image enhancement and image processing. Also call him xiaomai.",
  },
  {
    pic: githubPic("AsTheStarsFalll"),
    name: "AsTheStarsFall",
    intro: "None",
    url: "https://github.com/AsTheStarsFalll",
    note: "Focusing on semantic segmentation & attention mechanism. Some kind of nihilist.",
  },
  {
    pic: githubPic("breezeshane"),
    name: "Breeze Shane",
    intro: "一个专注理论但学不懂学不会的锈钢废物，但是他很擅长产出Bug，可能是因为他体表有源石结晶分布，但也可能仅仅是因为他是Bug本体。",
    url: "https://breezeshane.github.io/",
    note: "GANer, sometimes paranoid, we call him Old Shan.",
  },
  {
    pic: githubPic("AndPuQing"),
    name: "PuQing",
    intro: "intro * new",
    url: "https://github.com/AndPuQing",
    note: "Focusing on semantic segmentation, and a little frequency domain image processing.",
  },
];

function Friends() {
  const [friends, setFriends] = useState<FriendData[]>(friendsData);
  useEffect(() => {
    setFriends(arrayShuffle(friends));
  }, []);
  const [current, setCurrent] = useState(0);
  const [previous, setPrevious] = useState(0);
  useEffect(() => {
    // After `current` change, set a 300ms timer making `previous = current` so the previous card will be removed.
    const timer = setTimeout(() => {
      setPrevious(current);
    }, 300);

    return () => {
      // Before `current` change to another value, remove (possibly not triggered) timer, and make `previous = current`.
      clearTimeout(timer);
      setPrevious(current);
    };
  }, [current]);
  return (
    <div className="friends">
      <div style={{ position: "relative" }}>
        <div className="friend-columns">
          {/* Big card showing current selected */}
          <div className="friend-card-outer">
            {[
              previous != current && (
                <FriendCard key={previous} data={friends[previous]} fadeout />
              ),
              <FriendCard key={current} data={friends[current]} />,
            ]}
          </div>

          <div className="friend-list">
            {friends.map((x, i) => (
              <div
                key={x.name}
                className={clsx("friend-item", {
                  current: i == current,
                })}
                onClick={() => setCurrent(i)}
              >
                <img src={x.pic} alt="user profile photo" />
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function FriendCard(props: { data: FriendData; fadeout?: boolean }) {
  const { data, fadeout = false } = props;
  return (
    <div className={clsx("friend-card", { fadeout })}>
      <div className="card">
        <div className="card__image">
          <img
            src={data.pic}
            alt="User profile photo"
            title="User profile photo"
          />
        </div>
        <div className="card__body">
          <h2>{data.name}</h2>
          <p>
            <big>{data.intro}</big>
          </p>
          <p>
            <small>Comment : {data.note}</small>
          </p>
        </div>
        <div className="card__footer">
          <a href={data.url} className="button button--primary button--block">
            Visit
          </a>
        </div>
      </div>
    </div>
  );
}

export default About;
